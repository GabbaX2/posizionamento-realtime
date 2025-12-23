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
        # (Questo metodo rimane invariato rispetto alla versione con soglia 80%)
        try:
            data = request.get_json()
            curr_b64 = data.get('current_frame')
            session_id = data.get('session_id', 'default')
            current_step_req = str(data.get('current_step', 1))

            session = self.sessions.get(session_id)
            if not session: return {'success': False, 'error': 'Session Expired'}

            curr_img = self._decode_base64_image(curr_b64)
            ref_mask = self._decode_base64_image_gray(session['drawing_mask'])
            if curr_img is None or ref_mask is None: return {'success': False, 'error': 'Decode Error'}

            h_ref, w_ref = ref_mask.shape
            curr_img_resized = cv2.resize(curr_img, (w_ref, h_ref))

            # --- FASE 1: CALCOLO SCORE & CONTORNO (Nuova Logica Robusta) ---
            alignment_score, live_contour_points = self._calculate_alignment_and_contour(curr_img_resized, ref_mask)

            # Soglia Strict (80%)
            is_locked = alignment_score >= 80.0

            # --- FASE 2: SENSORI ---
            steps_config = session.get('steps_config', {})
            target_sensors = steps_config.get(current_step_req, [])
            sensor_status = []
            present_cnt = 0

            if steps_config:
                if is_locked:
                    for step_key, s_list in steps_config.items():
                        is_curr_step = (str(step_key) == current_step_req)
                        for s in s_list:
                            x, y, r = s['x'], s['y'], s['r']
                            x1, y1 = max(0, x - r), max(0, y - r)
                            x2, y2 = min(w_ref, x + r), min(h_ref, y + r)
                            roi = curr_img_resized[y1:y2, x1:x2]
                            is_present = check_sensor_color(roi, threshold=0.015)
                            sensor_status.append({'id': s['id'], 'step': int(step_key), 'present': is_present})
                            if is_curr_step and is_present: present_cnt += 1
                else:
                    for step_key, s_list in steps_config.items():
                        for s in s_list:
                            sensor_status.append({'id': s['id'], 'step': int(step_key), 'present': False})

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
                    direct = "Cerca di riempire il disegno"
                elif alignment_score < 80:
                    msg = "Quasi..."
                    direct = "Raddrizza e tieni fermo i bordi"
            else:
                if total_targets == 0:
                    final_acc = 100.0
                    msg = "Attesa Config"
                elif pct_sensors_found < 100:
                    final_acc = 80.0 + (pct_sensors_found * 0.19)
                    msg = "Allineato"
                    direct = f"Catturati {present_cnt}/{total_targets}"
                else:
                    final_acc = 100.0
                    msg = "Perfetto"
                    direct = "Step Completato!"

            return {
                'success': True,
                'accuracy': round(final_acc, 1),
                'is_locked': is_locked,
                'live_contour': live_contour_points,
                'message': msg,
                'direction': direct,
                'sensors_status': sensor_status
            }

        except Exception as e:
            logger.error(f'Validation Error: {e}')
            return {'success': False, 'error': str(e)}

    def _calculate_alignment_and_contour(self, live_img, ref_mask) -> Tuple[float, List]:
        """
        Nuova logica robusta:
        1. Preprocessing forte per ridurre rumore.
        2. Morfologia per unire i bordi in forme solide.
        3. Filtro: Sceglie il contorno più grande SOLO se è al centro dell'immagine.
        """
        h_img, w_img = live_img.shape[:2]
        img_center_x, img_center_y = w_img // 2, h_img // 2

        # 1. Preprocessing Robusto
        gray = cv2.cvtColor(live_img, cv2.COLOR_BGR2GRAY)
        # Sfocatura maggiore per sopprimere dettagli fini (piastrelle, texture)
        blurred = cv2.GaussianBlur(gray, (9, 9), 0)

        # Edge detection (Canny)
        edges = cv2.Canny(blurred, 30, 100)

        # 2. Operazioni Morfologiche (Cruciale!)
        # Usiamo una "Chiusura" (Dilatazione seguita da Erosione).
        # Questo serve a collegare linee interrotte e riempire piccoli buchi,
        # trasformando bordi confusi in una "forma" più solida.
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

        # Trova i contorni sull'immagine "pulita"
        contours, _ = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_contour = None
        max_area = 0

        # Definiamo una "Zona Centrale Sicura" (il 50% centrale dello schermo)
        margin_w = w_img * 0.25
        margin_h = h_img * 0.25
        safe_x_min, safe_x_max = margin_w, w_img - margin_w
        safe_y_min, safe_y_max = margin_h, h_img - margin_h

        # 3. Filtro di Centralità
        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Ignora oggetti troppo piccoli (es. meno del 2% dello schermo)
            if area < (w_img * h_img * 0.02): continue

            # Calcola il baricentro (centro di massa) del contorno
            M = cv2.moments(cnt)
            if M["m00"] == 0: continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            # Verifica se il centro del contorno cade nella zona sicura centrale
            is_central = (safe_x_min < cx < safe_x_max) and (safe_y_min < cy < safe_y_max)

            # Se è centrale ed è il più grande trovato finora, tienilo
            if is_central and area > max_area:
                max_area = area
                best_contour = cnt

        live_points = []
        score = 0.0

        if best_contour is not None:
            # Semplifica il contorno per l'invio al frontend
            epsilon = 0.004 * cv2.arcLength(best_contour, True)
            approx = cv2.approxPolyDP(best_contour, epsilon, True)
            live_points = approx.reshape(-1, 2).tolist()

            # --- CALCOLO SCORE (Tollerante per soglia 80%) ---
            ref_edges_ref = cv2.Canny(ref_mask, 50, 150)
            # Tolleranza elevata (circa 10px per lato) per facilitare il raggiungimento dell'80%
            ref_fat = cv2.dilate(ref_edges_ref, np.ones((21, 21), np.uint8))

            live_edges_mask = np.zeros_like(ref_mask)
            cv2.drawContours(live_edges_mask, [best_contour], -1, 255, thickness=2)

            intersection = cv2.bitwise_and(live_edges_mask, live_edges_mask, mask=ref_fat)
            matched_pixels = cv2.countNonZero(intersection)
            total_live_pixels = cv2.countNonZero(live_edges_mask)

            if total_live_pixels > 0:
                raw_score = (matched_pixels / total_live_pixels) * 100
                score = min(100, raw_score)
        else:
            # Nessun contorno valido trovato al centro
            score = 0.0

        return score, live_points

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
