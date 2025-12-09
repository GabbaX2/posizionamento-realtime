import cv2
import base64
import numpy as np
import os
import logging
from typing import Dict, List

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

    def _remove_overlapping_circles(self, circles, min_dist):
        if not circles: return []
        clean = []
        for c in circles:
            x, y, r = c
            is_overlap = False
            for ex, ey, er in clean:
                dist = np.sqrt((x - ex) ** 2 + (y - ey) ** 2)
                if dist < min_dist:
                    is_overlap = True;
                    break
            if not is_overlap: clean.append(c)
        return clean

    # --- VALIDAZIONE LIVE (Logica Edge-Based)
    def validate_current_frame(self, request) -> Dict:
        try:
            data = request.get_json()
            curr_b64 = data.get('current_frame')
            session_id = data.get('session_id', 'default')
            current_step_req = str(data.get('current_step', 1))

            session = self.sessions.get(session_id)
            if not session: return {'success': False, 'error': 'Session Expired'}

            # 1. Decodifica frame live (webcam)
            curr_img = self._decode_base64_image(curr_b64)
            if curr_img is None: return {'success': False, 'error': 'Frame Error'}

            # 2. Decodifica maschera di riferimento usando grayscale perchè maschera bianria
            ref_mask = self._decode_base64_image_gray(session['drawing_mask'])

            # 3. Ridimensionamento
            h_ref, w_ref = ref_mask.shape
            curr_img_resized = cv2.resize(curr_img, (w_ref, h_ref))

            # VERIFICA ALLINEAMENTO
            # Confronta i bordi estratti dalla webcam con il disegno di riferimento
            alignment_score = self._check_alignment_edges(curr_img_resized, ref_mask)

            # Se la soglia di allineamento è <40%, significa che la gamba non è sovrapposta al disegno.
            is_aligned = alignment_score > 40.0

            # VERIFICA SENSORI (solo se la gamba è allineata)
            sensor_status = []
            steps_config = session['step_config']
            present_cnt = 0
            target_sensors = steps_config.get(current_step_req, [])

            if is_aligned:
                # se l'utente è allineato col disegno, controllo sensori
                for step_key, s_list in steps_config.items():
                    is_curr_step = (str(step_key) == current_step_req)
                    for s in s_list:
                        # Estraiamo la ROI (region of interest)
                        x, y, r = s['x'], s['y'], s['r']

                        # Calcolo coordinate
                        x1, y1 = max(0, x - r), max(0, y - r)
                        x2, y2 = min(w_ref, x + r), min(h_ref, y + r)
                        roi = curr_img_resized[y1:y2, x1:x2]

                        # controlliamo il Colore dell'elettrodo (Azzurro/Grigio)
                        is_present = check_sensor_color(roi, threshold=0.015)

                        sensor_status.append({'id': s['id'], 'step': int(step_key), 'present': is_present})
                        if is_curr_step and is_present: present_cnt += 1
            else:
                # Se non è allineato, stato: falso
                for step_key, s_list in steps_config.items():
                    for s in s_list:
                        sensor_status.append({'id': s['id'], 'step': int(step_key), 'present': False})

            # --- CALCOLO PUNTEGGIO E MESSAGGI UI ---
            total_targets = len(target_sensors)
            sensors_score = (present_cnt / total_targets * 100) if total_targets > 0 else 0

            final_acc = 0
            msg = ""
            direct = ""

            if not is_aligned:
                # Caso 1: Utente non posizionato correttamente
                final_acc = alignment_score  # Mostriamo quanto è lontano
                msg = "Non allineato"
                direct = "Sovrapponi la gamba al disegno"
            elif sensors_score < 100:
                # Caso 2: Allineato bene, ma mancano sensori
                # Score parte da 50% + bonus sensori
                final_acc = 70 + (sensors_score * 0.25)
                msg = "Controllo Sensori"
                direct = f"Mancano {total_targets - present_cnt} sensori"
            else:
                # Caso 3: Tutto perfetto
                final_acc = 98.00
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
    """
        Confronta i bordi della webcam (indipendentemente dal colore/sfondo)
        con il disegno di riferimento.
     """
    # 1. Converti Webcam in scala di grigi
    gray = cv2.cvtColor(live_img_color, cv2.COLOR_BGR2GRAY)

    # 2. Sfoca leggermente per rimuovere rumore video
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 3. Canny Edge Detection: Trova i contorni strutturali della gamba
    # Soglie 50-150 sono standard per ambienti interni
    live_edges = cv2.Canny(blurred, 50, 150)

    # 4. Dilatazione del riferimento (TOLLERANZA)
    # Il disegno è una linea sottile (1px o 3px). Difficile sovrapporsi perfettamente
    # Ingrassiamo la linea di riferimento di 5-7 pixel
    # Sel il bordo della webcam rientra dentro questa zona, vale come punto
    kernel = np.ones((5, 5), np.uint8)
    ref_mask_dilated = cv2.dilate(ref_mask_gray, kernel, iterations=1)

    # 5. Sovrapposizione (AND Logico)
    # Pixel bianchi (live edge) che cadono su pixel bianchi (ref mask dilatata)
    overlap = cv2.bitwise_and(live_edges, live_edges, mask=ref_mask_dilated)

    # 6. Calcolo Score
    live_edges_pixels = cv2.countNonZero(live_edges)
    if live_edges_pixels == 0: return 0.0

    matching_pixels = cv2.countNonZero(overlap)

    # Percentuale di bordi live che sono corretti
    score = (matching_pixels / live_edges_pixels) * 100

    adjusted_score = min(100, score * 3.0)

    return adjusted_score

    # Helpers Base64


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
