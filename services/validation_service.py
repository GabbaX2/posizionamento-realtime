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
        """Gestisce upload reference con overlay semi-trasparente e lista sensori."""
        try:
            if 'file' not in request.files: 
                return {'success': False, 'error': 'Nessun file'}
            
            result = self.processor.process(request.files['file'])
            if not result['success']: 
                return result
            
            session_id = request.form.get('session_id', 'default')
            
            # Salva overlay con alpha e lista sensori per concentricità
            self.sessions[session_id] = {
                'drawing_mask': result['drawing_mask_b64'],
                'drawing_overlay': result.get('drawing_overlay_b64', result['drawing_mask_b64']),
                'sensor_circles': result.get('sensor_circles', []),  # Lista flat di sensori
                'steps_config': result['steps_config'],
                'dims': result['dims']
            }
            
            return {
                'success': True,
                'session_id': session_id,
                'reference_image': result['reference_clean_b64'],
                'drawing_overlay_image': result.get('drawing_overlay_b64', result['drawing_mask_b64']),
                'steps_config': result['steps_config'],
                'sensor_circles': result.get('sensor_circles', []),
                'total_steps': len(result['steps_config'])
            }
        except Exception as e:
            logger.error(f'Upload Error: {e}')
            return {'success': False, 'error': str(e)}

    def validate_current_frame(self, request) -> Dict:
        """
        Sistema migliorato a 2 fasi:
        1. Template Matching: Confronta overlay semi-trasparente con frame live
        2. HoughCircles: Rileva cerchi fisici e verifica concentricità
        """
        try:
            data = request.get_json()
            curr_b64 = data.get('current_frame')
            session_id = data.get('session_id', 'default')
            current_step_req = str(data.get('current_step', 1))

            session = self.sessions.get(session_id)
            if not session: 
                return {'success': False, 'error': 'Session Expired'}

            curr_img = self._decode_base64_image(curr_b64)
            ref_overlay = self._decode_base64_image(session['drawing_mask'])  # Usa overlay BGR
            
            if curr_img is None or ref_overlay is None: 
                return {'success': False, 'error': 'Decode Error'}

            h_ref, w_ref = ref_overlay.shape[:2]
            curr_img_resized = cv2.resize(curr_img, (w_ref, h_ref))

            # --- FASE 1: TEMPLATE MATCHING PER ZONA ---
            is_aligned, zone_match_score = self._check_zone_alignment(curr_img_resized, ref_overlay)
            
            # Converti in percentuale per feedback (score va da -1 a 1, normalizziamo 0.3-0.7 -> 0-100%)
            alignment_pct = max(0, min(100, (zone_match_score - 0.3) / 0.4 * 100))
            
            # Lock si attiva sopra soglia 0.6 (60% correlazione)
            is_locked = is_aligned
            
            # --- FASE 2: RILEVAMENTO CERCHI E CONCENTRICITÀ (solo se locked) ---
            sensor_circles_ref = session.get('sensor_circles', [])
            steps_config = session.get('steps_config', {})
            target_sensors = steps_config.get(current_step_req, [])
            
            sensor_status = []
            present_cnt = 0
            missing_cnt = 0
            group_feedback = ""
            
            if is_locked and sensor_circles_ref:
                # Rileva e verifica sensori
                sensor_results = self._detect_and_verify_sensors(curr_img_resized, sensor_circles_ref)
                
                # Genera status per ogni sensore
                for step_key, s_list in steps_config.items():
                    is_curr_step = (str(step_key) == current_step_req)
                    for s in s_list:
                        sensor_id = s['id']
                        result = sensor_results.get(sensor_id, {'present': False, 'group': int(step_key)})
                        
                        sensor_status.append({
                            'id': sensor_id,
                            'step': int(step_key),
                            'present': result['present'],
                            'in_area': True  # Se locked, assume dentro area
                        })
                        
                        if is_curr_step:
                            if result['present']:
                                present_cnt += 1
                            else:
                                missing_cnt += 1
                
                # Genera feedback per gruppo corrente
                group_feedback = self._generate_group_feedback(sensor_results, steps_config, current_step_req)
            else:
                # Non locked: tutti i sensori non presenti
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
                final_acc = alignment_pct
                if alignment_pct < 20:
                    msg = "Inquadra la gamba"
                    direct = "Posiziona il soggetto al centro"
                elif alignment_pct < 50:
                    msg = "Avvicinati..."
                    direct = f"Sovrapposizione: {int(alignment_pct)}%"
                elif alignment_pct < 80:
                    msg = "Quasi..."
                    direct = "Centra e sovrapponi l'overlay"
                else:
                    msg = "Blocco imminente"
                    direct = f"Match: {int(zone_match_score*100)}%"
            else:
                if total_targets == 0:
                    final_acc = 100.0
                    msg = "Zona Validata"
                    direct = "Nessun sensore configurato"
                elif pct_sensors_found < 100:
                    final_acc = 80.0 + (pct_sensors_found * 0.19)
                    msg = "Zona OK"
                    direct = group_feedback if group_feedback else f"Sensori: {present_cnt}/{total_targets}"
                else:
                    final_acc = 100.0
                    msg = "Perfetto"
                    direct = "Gruppo completo! ✅"

            return {
                'success': True,
                'accuracy': round(final_acc, 1),
                'is_locked': is_locked,
                'zone_match': round(zone_match_score * 100, 1),
                'message': msg,
                'direction': direct,
                'group_feedback': group_feedback,
                'sensors_status': sensor_status
            }

        except Exception as e:
            logger.error(f'Validation Error: {e}')
            return {'success': False, 'error': str(e)}
    
    def _check_zone_alignment(self, live_frame, reference_overlay):
        """
        Usa template matching per verificare che la gamba sia nella zona giusta.
        L'overlay include la gamba semi-trasparente come "template".
        """
        # Converti a grayscale per matching
        gray_live = cv2.cvtColor(live_frame, cv2.COLOR_BGR2GRAY)
        gray_ref = cv2.cvtColor(reference_overlay, cv2.COLOR_BGR2GRAY)
        
        # Per template matching, serve che il template sia più piccolo dell'immagine
        # Usiamo una finestra centrale del reference come template
        h, w = gray_ref.shape
        margin_h, margin_w = h // 8, w // 8
        template = gray_ref[margin_h:h-margin_h, margin_w:w-margin_w]
        
        if template.size == 0:
            return False, 0.0
        
        # Template matching con metodo normalizado
        result = cv2.matchTemplate(gray_live, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)
        
        # Soglia: 0.6 = 60% match
        is_aligned = max_val >= 0.55
        return is_aligned, max_val
    
    def _detect_and_verify_sensors(self, live_frame, sensor_circles_ref: List[Dict]) -> Dict:
        """
        1. Rileva cerchi nel frame live con HoughCircles
        2. Per ogni cerchio di riferimento, verifica se c'è un cerchio live concentrico
        3. Concentrico = distanza tra centri < soglia (es. 25px)
        """
        gray = cv2.cvtColor(live_frame, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.medianBlur(gray, 5)
        
        # Parametri HoughCircles ottimizzati per sensori
        circles_live = cv2.HoughCircles(
            gray_blur, cv2.HOUGH_GRADIENT, 
            dp=1.2,
            minDist=40, 
            param1=50, 
            param2=30,
            minRadius=15, 
            maxRadius=50
        )
        
        results = {}
        
        for ref in sensor_circles_ref:
            ref_id = ref['id']
            ref_x, ref_y = ref['x'], ref['y']
            found = False
            
            if circles_live is not None:
                for circle in circles_live[0]:
                    x, y, r = circle
                    dist = np.sqrt((x - ref_x)**2 + (y - ref_y)**2)
                    if dist < 25:  # Soglia concentricità: 25px
                        found = True
                        break
            
            results[ref_id] = {
                'present': found,
                'group': ref.get('group', 1)
            }
        
        return results
    
    def _generate_group_feedback(self, sensor_results: Dict, steps_config: Dict, current_step: str) -> str:
        """
        Per ogni gruppo (step), conta quanti sensori mancano e genera feedback specifico.
        """
        step_sensors = steps_config.get(current_step, [])
        
        if not step_sensors:
            return ""
        
        missing = sum(1 for s in step_sensors if not sensor_results.get(s['id'], {}).get('present', False))
        total = len(step_sensors)
        
        if missing == 0:
            return "Gruppo completo! ✅"
        elif missing == 1:
            return "Manca 1 sensore nel gruppo"
        elif missing == total:
            return f"Posizionare tutti e {total} i sensori"
        else:
            return f"Mancano {missing} sensori, posizionarli"

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
