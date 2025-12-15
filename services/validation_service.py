import cv2
import base64
import numpy as np
import logging
from typing import Dict

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

            # 2. Resize
            h_ref, w_ref = ref_mask.shape
            curr_img_resized = cv2.resize(curr_img, (w_ref, h_ref))

            # 3. VERIFICA ALLINEAMENTO (Logica Anti-Pavimento)
            alignment_score = self._check_alignment_edges(curr_img_resized, ref_mask)

            # Soglia: 25% (Accessibile per la gamba, impossibile per il pavimento grazie al nuovo algoritmo)
            is_aligned = alignment_score > 25.0

            # 4. SENSORI
            steps_config = session.get('steps_config', {})
            target_sensors = steps_config.get(current_step_req, [])
            sensor_status = []
            present_cnt = 0

            if steps_config:
                if is_aligned:
                    for step_key, s_list in steps_config.items():
                        is_curr_step = (str(step_key) == current_step_req)
                        for s in s_list:
                            x, y, r = s['x'], s['y'], s['r']
                            x1, y1 = max(0, x - r), max(0, y - r)
                            x2, y2 = min(w_ref, x + r), min(h_ref, y + r)
                            roi = curr_img_resized[y1:y2, x1:x2]

                            is_present = check_sensor_color(roi, threshold=0.015)
                            sensor_status.append({'id': s['id'], 'step': int(step_key), 'present': is_present})

                            if is_curr_step and is_present:
                                present_cnt += 1
                else:
                    for step_key, s_list in steps_config.items():
                        for s in s_list:
                            sensor_status.append({'id': s['id'], 'step': int(step_key), 'present': False})

            # 5. UI FEEDBACK
            total_targets = len(target_sensors)
            pct_sensors_found = (present_cnt / total_targets * 100) if total_targets > 0 else 0

            final_acc = 0
            msg = "..."
            direct = "..."

            if not is_aligned:
                final_acc = alignment_score
                # Feedback specifico se lo score è 0 (filtro anti-pavimento attivo)
                if alignment_score < 5:
                    msg = "Oggetto errato"
                    direct = "Inquadra solo la gamba"
                else:
                    msg = "Migliora Allineamento"
                    direct = "Segui i bordi del disegno"
            else:
                if total_targets == 0:
                    final_acc = 80.0
                    msg = "Allineato"
                    direct = "Attendo config..."
                elif pct_sensors_found < 100:
                    # Parte da 50 e sale a 98
                    final_acc = 50.0 + (pct_sensors_found * 0.48)
                    msg = "Controllo Sensori"
                    direct = f"Trovati {present_cnt} su {total_targets}"
                else:
                    final_acc = 99.0
                    msg = "Perfetto"
                    direct = "Tieni fermo"

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
        Logica "Anti-Pavimento":
        Confronta i bordi interni con quelli esterni.
        Se l'esterno è caotico quanto l'interno, restituisce 0.
        """
        # 1. Edge Detection
        gray = cv2.cvtColor(live_img_color, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        # Soglie basse per vedere la pelle
        live_edges = cv2.Canny(blurred, 30, 100)

        # 2. Creazione Zone
        kernel = np.ones((5, 5), np.uint8)

        # A) Zona Interna (Disegno)
        mask_in = ref_mask_gray

        # B) Zona Esterna "Lontana" (Rumore)
        # Dilatiamo di ~40px. Tutto ciò che è fuori è sfondo puro.
        mask_dilated = cv2.dilate(ref_mask_gray, kernel, iterations=20)
        mask_outside = cv2.bitwise_not(mask_dilated)

        # 3. Conteggi
        edges_in = cv2.countNonZero(cv2.bitwise_and(live_edges, live_edges, mask=mask_in))
        edges_out = cv2.countNonZero(cv2.bitwise_and(live_edges, live_edges, mask=mask_outside))

        # --- FILTRO 1: Densità Minima ---
        # Se dentro è tutto nero (muro bianco, foglio), esci.
        area_in = cv2.countNonZero(mask_in)
        if area_in == 0: return 0.0
        if (edges_in / area_in) < 0.005:
            return 0.0  # Troppo liscio/vuoto

        # --- FILTRO 2: KILL-SWITCH PAVIMENTO ---
        # Questa è la chiave. Un pavimento ha bordi ovunque. Una gamba ha bordi dentro e sfondo vuoto fuori.
        # Se i bordi fuori sono più numerosi dei bordi dentro, stai inquadrando una texture uniforme (pavimento).
        if edges_out > edges_in:
            return 0.0  # BOCCIATURA IMMEDIATA

        # --- Calcolo Punteggio (Solo se superi i filtri) ---

        # 1. Match sui bordi del disegno (Contour Matching)
        # Creiamo il contorno del disegno (la linea nera)
        ref_edges = cv2.Canny(ref_mask_gray, 50, 150)
        ref_edges_fat = cv2.dilate(ref_edges, kernel, iterations=2)  # Spessore tolleranza

        # Quanti bordi live cadono ESATTAMENTE sulla linea del disegno?
        matching_pixels = cv2.countNonZero(cv2.bitwise_and(live_edges, live_edges, mask=ref_edges_fat))
        total_ref_pixels = cv2.countNonZero(ref_edges)

        if total_ref_pixels == 0: return 0.0

        # Punteggio basato su quanto "ricalchi" il disegno
        contour_score = (matching_pixels / total_ref_pixels) * 100

        # Moltiplichiamo per un fattore (es. 2.5) perché non ricalcherai mai al 100%
        final_score = min(100, contour_score * 2.5)

        return final_score

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
