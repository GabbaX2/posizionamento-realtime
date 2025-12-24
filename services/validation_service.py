import cv2
import base64
import numpy as np
import logging
from typing import Dict, List, Tuple

from processors.reference_processor import ReferenceProcessor
from utils.sensor_utils import check_sensor_color

logger = logging.getLogger(__name__)


class ValidationService:
    def __init__(self):
        self.sessions = {}
        self.processor = ReferenceProcessor()

    def handle_reference_upload(self, request):
        # (Questo metodo rimane invariato)
        try:
            if 'file' not in request.files: return {'success': False, 'error': 'Nessun file'}
            result = self.processor.process(request.files['file'])
            if not result['success']: return result
            session_id = request.form.get('session_id', 'default')
            self.sessions[session_id] = {
                'drawing_mask': result['drawing_mask_b64'],
                'steps_config': result['steps_config'],
                'dims': result['dims']
            }
            return {
                'success': True,
                'session_id': session_id,
                'reference_image': result['reference_clean_b64'],
                'drawing_overlay_image': result['drawing_mask_b64'],
                'steps_config': result['steps_config'],
                'total_steps': len(result['steps_config'])
            }
        except Exception as e:
            logger.error(f'Upload Error: {e}')
            return {'success': False, 'error': str(e)}

    def validate_current_frame(self, request) -> Dict:
        """
        Sistema migliorato a 3 fasi:
        1. Estrae contorno preciso dal frame live
        2. Confronta con il contorno di riferimento usando shape matching
        3. Solo se combaciano, attiva il lock e verifica i sensori nell'area designata
        """
        try:
            data = request.get_json()
            curr_b64 = data.get('current_frame')
            session_id = data.get('session_id', 'default')
            current_step_req = str(data.get('current_step', 1))

            session = self.sessions.get(session_id)
            if not session: return {'success': False, 'error': 'Session Expired'}

            curr_img = self._decode_base64_image(curr_b64)
            ref_mask = self._decode_base64_image_gray(session['drawing_mask'])
            ref_contour = session.get('reference_contour')
            
            if curr_img is None or ref_mask is None: 
                return {'success': False, 'error': 'Decode Error'}

            h_ref, w_ref = ref_mask.shape
            curr_img_resized = cv2.resize(curr_img, (w_ref, h_ref))

            # --- FASE 1: ESTRAZIONE CONTORNO LIVE PRECISO ---
            live_contour, live_contour_points = self._extract_live_contour(curr_img_resized)
            
            # --- FASE 2: CONFRONTO CONTORNI (SHAPE MATCHING) ---
            shape_similarity = 1.0  # Default: non simili
            alignment_score = 0.0
            
            if live_contour is not None and ref_contour is not None:
                # Usa cv2.matchShapes per confrontare le forme
                # Valore < 0.1 = molto simili, > 0.3 = molto diversi
                shape_similarity = cv2.matchShapes(live_contour, ref_contour, cv2.CONTOURS_MATCH_I1, 0)
                
                # Calcola anche uno score visivo per feedback utente
                alignment_score = self._calculate_visual_alignment_score(
                    live_contour, ref_contour, (h_ref, w_ref)
                )
            
            # LOCK si attiva solo se i contorni combaciano bene
            # Soglia shape matching: 0.15 (più basso = più preciso)
            is_locked = (shape_similarity < 0.15) and (alignment_score >= 75.0)

            # --- FASE 3: VERIFICA SENSORI (solo se locked) ---
            steps_config = session.get('steps_config', {})
            target_sensors = steps_config.get(current_step_req, [])
            sensor_status = []
            present_cnt = 0

            if steps_config:
                if is_locked and live_contour is not None:
                    # Verifica sensori SOLO nell'area del contorno validato
                    for step_key, s_list in steps_config.items():
                        is_curr_step = (str(step_key) == current_step_req)
                        for s in s_list:
                            x, y, r = s['x'], s['y'], s['r']
                            
                            # Controlla se il sensore è dentro il contorno
                            point_in_contour = cv2.pointPolygonTest(live_contour, (x, y), False) >= 0
                            
                            is_present = False
                            if point_in_contour:
                                # Solo se è dentro il contorno, verifica il colore
                                x1, y1 = max(0, x - r), max(0, y - r)
                                x2, y2 = min(w_ref, x + r), min(h_ref, y + r)
                                roi = curr_img_resized[y1:y2, x1:x2]
                                is_present = check_sensor_color(roi, threshold=0.015)
                            
                            sensor_status.append({
                                'id': s['id'], 
                                'step': int(step_key), 
                                'present': is_present,
                                'in_area': point_in_contour
                            })
                            if is_curr_step and is_present: 
                                present_cnt += 1
                else:
                    for step_key, s_list in steps_config.items():
                        for s in s_list:
                            sensor_status.append({
                                'id': s['id'], 
                                'step': int(step_key), 
                                'present': False,
                                'in_area': False
                            })

            # --- UI FEEDBACK ---
            total_targets = len(target_sensors)
            pct_sensors_found = (present_cnt / total_targets * 100) if total_targets > 0 else 0
            final_acc = 0.0
            msg = "..."
            direct = "..."

            if not is_locked:
                final_acc = alignment_score
                if alignment_score < 10:
                    msg = "Inquadra la gamba"
                    direct = "Posiziona il soggetto al centro"
                elif alignment_score < 50:
                    msg = "Avvicinati..."
                    direct = f"Contorno rilevato - allinealo (match: {int((1-shape_similarity)*100)}%)"
                elif alignment_score < 75:
                    msg = "Quasi..."
                    direct = "Centra e sovrapponi il disegno"
            else:
                if total_targets == 0:
                    final_acc = 100.0
                    msg = "Contorno Validato"
                    direct = "Attesa Config Sensori"
                elif pct_sensors_found < 100:
                    final_acc = 80.0 + (pct_sensors_found * 0.19)
                    msg = "Contorno OK"
                    direct = f"Posiziona sensori: {present_cnt}/{total_targets}"
                else:
                    final_acc = 100.0
                    msg = "Perfetto"
                    direct = "Step Completato!"

            return {
                'success': True,
                'accuracy': round(final_acc, 1),
                'is_locked': is_locked,
                'live_contour': live_contour_points,
                'shape_match': round((1 - shape_similarity) * 100, 1),  # % similarità
                'message': msg,
                'direction': direct,
                'sensors_status': sensor_status
            }

        except Exception as e:
            logger.error(f'Validation Error: {e}')
            return {'success': False, 'error': str(e)}

    def _extract_reference_contour(self, ref_mask):
        """
        Estrae il contorno principale dal disegno di riferimento.
        Restituisce il contorno più grande come riferimento preciso.
        """
        try:
            # Applica edge detection sulla mask
            edges = cv2.Canny(ref_mask, 50, 150)
            
            # Trova contorni
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return None
            
            # Prendi il contorno più grande (dovrebbe essere la gamba)
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Semplifica leggermente per renderlo più robusto al matching
            epsilon = 0.002 * cv2.arcLength(largest_contour, True)
            approx_contour = cv2.approxPolyDP(largest_contour, epsilon, True)
            
            logger.info(f"✅ Contorno di riferimento estratto: {len(approx_contour)} punti")
            return approx_contour
            
        except Exception as e:
            logger.error(f"Errore estrazione contorno riferimento: {e}")
            return None

    def _extract_live_contour(self, live_img):
        """
        Estrae il contorno preciso dalla camera live.
        Preprocessing robusto per gestire rumore e dettagli.
        Restituisce: (contour_object, points_list_for_frontend)
        """
        h_img, w_img = live_img.shape[:2]
        img_center_x, img_center_y = w_img // 2, h_img // 2

        # 1. Preprocessing Robusto
        gray = cv2.cvtColor(live_img, cv2.COLOR_BGR2GRAY)
        # Sfocatura maggiore per sopprimere dettagli fini
        blurred = cv2.GaussianBlur(gray, (9, 9), 0)

        # Edge detection
        edges = cv2.Canny(blurred, 30, 100)

        # 2. Operazioni Morfologiche per connettere bordi
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

        # Trova i contorni
        contours, _ = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_contour = None
        max_area = 0

        # Zona Centrale Sicura (50% centrale dello schermo)
        margin_w = w_img * 0.25
        margin_h = h_img * 0.25
        safe_x_min, safe_x_max = margin_w, w_img - margin_w
        safe_y_min, safe_y_max = margin_h, h_img - margin_h

        # 3. Filtro di Centralità - prende solo contorni centrali
        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Ignora oggetti troppo piccoli (< 2% dello schermo)
            if area < (w_img * h_img * 0.02): 
                continue

            # Calcola il baricentro
            M = cv2.moments(cnt)
            if M["m00"] == 0: 
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            # Verifica centralità
            is_central = (safe_x_min < cx < safe_x_max) and (safe_y_min < cy < safe_y_max)

            if is_central and area > max_area:
                max_area = area
                best_contour = cnt

        live_points = []
        
        if best_contour is not None:
            # Semplifica per l'invio al frontend
            epsilon = 0.004 * cv2.arcLength(best_contour, True)
            approx = cv2.approxPolyDP(best_contour, epsilon, True)
            live_points = approx.reshape(-1, 2).tolist()
            return best_contour, live_points
        else:
            return None, []

    def _calculate_visual_alignment_score(self, live_contour, ref_contour, img_dims):
        """
        Calcola uno score visivo (0-100) per dare feedback all'utente.
        Basato su sovrapposizione dei contorni.
        """
        try:
            h, w = img_dims
            
            # Crea maschere dai contorni
            ref_mask = np.zeros((h, w), dtype=np.uint8)
            live_mask = np.zeros((h, w), dtype=np.uint8)
            
            cv2.drawContours(ref_mask, [ref_contour], -1, 255, thickness=cv2.FILLED)
            cv2.drawContours(live_mask, [live_contour], -1, 255, thickness=cv2.FILLED)
            
            # Calcola intersezione e unione
            intersection = cv2.bitwise_and(ref_mask, live_mask)
            union = cv2.bitwise_or(ref_mask, live_mask)
            
            intersection_area = cv2.countNonZero(intersection)
            union_area = cv2.countNonZero(union)
            
            if union_area == 0:
                return 0.0
            
            # IoU (Intersection over Union) come score
            iou = (intersection_area / union_area) * 100
            
            return min(100.0, iou)
            
        except Exception as e:
            logger.error(f"Errore calcolo alignment score: {e}")
            return 0.0


    # (Metodi di decodifica helper rimangono invariati)
    def _decode_base64_image(self, b64):
        if not b64: return None
        if ',' in b64: b64 = b64.split(',')[1]
        try:
            return cv2.imdecode(np.frombuffer(base64.b64decode(b64), np.uint8), cv2.IMREAD_COLOR)
        except:
            return None

    def _decode_base64_image_gray(self, b64):
        if not b64: return None
        if ',' in b64: b64 = b64.split(',')[1]
        try:
            return cv2.imdecode(np.frombuffer(base64.b64decode(b64), np.uint8), cv2.IMREAD_GRAYSCALE)
        except:
            return None
