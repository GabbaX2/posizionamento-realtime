import cv2
import base64
import numpy as np
import logging
from typing import Dict

# Assicurati che i path siano corretti per il tuo progetto
from processors.reference_processor import ReferenceProcessor
from utils.sensor_utils import check_sensor_color

logger = logging.getLogger(__name__)


class ValidationService:
    def __init__(self):
        self.sessions = {}
        self.processor = ReferenceProcessor()

    def handle_reference_upload(self, request):
        try:
            if 'file' not in request.files: return {'success': False, 'error': 'Nessun file'}

            result = self.processor.process(request.files['file'])

            if not result['success']:
                return result

            session_id = request.form.get('session_id', 'default')

            # Salvataggio in memoria (Attenzione: si perde se riavvii il server!)
            self.sessions[session_id] = {
                'drawing_mask': result['drawing_mask_b64'],
                'steps_config': result['steps_config'],  # Qui salviamo con la 's'
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
        try:
            data = request.get_json()
            curr_b64 = data.get('current_frame')
            session_id = data.get('session_id', 'default')
            current_step_req = str(data.get('current_step', 1))

            session = self.sessions.get(session_id)
            if not session:
                return {'success': False, 'error': 'Session Expired'}

            # 1. Decodifica Immagini
            curr_img = self._decode_base64_image(curr_b64)
            ref_mask = self._decode_base64_image_gray(session['drawing_mask'])

            if curr_img is None or ref_mask is None:
                return {'success': False, 'error': 'Image Decode Error'}

            # 2. Ridimensionamento dinamico (Webcam -> Reference)
            h_ref, w_ref = ref_mask.shape
            curr_img_resized = cv2.resize(curr_img, (w_ref, h_ref))

            # 3. VERIFICA ALLINEAMENTO (Bordi)
            alignment_score = self._check_alignment_edges(curr_img_resized, ref_mask)
            is_aligned = alignment_score > 40.0

            # 4. VERIFICA SENSORI
            # CORRETTO: Uso 'steps_config' (plurale) come salvato nell'upload
            steps_config = session.get('steps_config', {})
            target_sensors = steps_config.get(current_step_req, [])

            sensor_status = []
            present_cnt = 0

            # Controlla i sensori solo se c'è una configurazione
            if steps_config:
                if is_aligned:
                    # Se l'utente è allineato, controlliamo i sensori specifici dello step
                    for step_key, s_list in steps_config.items():
                        is_curr_step = (str(step_key) == current_step_req)
                        for s in s_list:
                            # Estrai ROI (Region of Interest)
                            x, y, r = s['x'], s['y'], s['r']
                            x1, y1 = max(0, x - r), max(0, y - r)
                            x2, y2 = min(w_ref, x + r), min(h_ref, y + r)
                            roi = curr_img_resized[y1:y2, x1:x2]

                            # Check Colore
                            is_present = check_sensor_color(roi, threshold=0.015)
                            sensor_status.append({'id': s['id'], 'step': int(step_key), 'present': is_present})

                            if is_curr_step and is_present:
                                present_cnt += 1
                else:
                    # Se NON è allineato, segna tutto come falso
                    for step_key, s_list in steps_config.items():
                        for s in s_list:
                            sensor_status.append({'id': s['id'], 'step': int(step_key), 'present': False})

            # 5. CALCOLO MESSAGGI UI
            total_targets = len(target_sensors)
            sensors_score = (present_cnt / total_targets * 100) if total_targets > 0 else 0

            final_acc = 0
            msg = "..."
            direct = "..."

            if not is_aligned:
                final_acc = alignment_score
                msg = "Non allineato"
                direct = "Sovrapponi la gamba al disegno"
            elif sensors_score < 100:
                final_acc = 70 + (sensors_score * 0.25)
                msg = "Controllo Sensori"
                direct = f"Mancano {total_targets - present_cnt} sensori"
            else:
                final_acc = 98.0
                msg = "Perfetto"
                direct = "Tieni fermo per confermare"

            return {
                'success': True,
                'accuracy': round(final_acc, 1),
                'message': msg,
                'direction': direct,
                'sensors_status': sensor_status,
                'debug_alignment': alignment_score
            }

        except Exception as e:
            logger.error(f'Validation Error: {e}')
            return {'success': False, 'error': str(e)}

    def _check_alignment_edges(self, live_img_color, ref_mask_gray):
        """Confronta i bordi webcam con maschera riferimento"""
        gray = cv2.cvtColor(live_img_color, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        live_edges = cv2.Canny(blurred, 50, 150)

        # Tolleranza: dilata il riferimento
        kernel = np.ones((5, 5), np.uint8)
        ref_mask_dilated = cv2.dilate(ref_mask_gray, kernel, iterations=1)

        overlap = cv2.bitwise_and(live_edges, live_edges, mask=ref_mask_dilated)

        live_edges_pixels = cv2.countNonZero(live_edges)
        if live_edges_pixels == 0: return 0.0

        matching_pixels = cv2.countNonZero(overlap)
        score = (matching_pixels / live_edges_pixels) * 100

        # Boost dello score per UX
        return min(100, score * 3.0)

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
