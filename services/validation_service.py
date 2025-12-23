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
        try:
            if 'file' not in request.files:
                return {'success': False, 'error': 'Nessun file'}

            result = self.processor.process(request.files['file'])

            if not result['success']:
                return result

            session_id = request.form.get('session_id', 'default')

            self.sessions[session_id] = {
                'drawing_mask': result['drawing_mask_b64'],  # Immagine binaria del disegno
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

            # 2. Resize immagine live per combaciare con le dimensioni del riferimento
            h_ref, w_ref = ref_mask.shape
            curr_img_resized = cv2.resize(curr_img, (w_ref, h_ref))

            # 3. FASE 1: CALCOLO ALLINEAMENTO E ESTRAZIONE CONTORNO
            # Restituisce lo score (0-100) e i punti XY del contorno live per il frontend
            alignment_score, live_contour_points = self._calculate_alignment_and_contour(curr_img_resized, ref_mask)

            # Soglia di Lock: definisce quando l'utente è "dentro"
            # Se > 40%, consideriamo l'allineamento valido e passiamo ai sensori
            is_locked = alignment_score > 40.0

            # 4. FASE 2: SENSORI (Solo se LOCKED)
            steps_config = session.get('steps_config', {})
            target_sensors = steps_config.get(current_step_req, [])
            sensor_status = []
            present_cnt = 0

            if steps_config:
                if is_locked:
                    # L'utente è allineato, processiamo i sensori
                    for step_key, s_list in steps_config.items():
                        is_curr_step = (str(step_key) == current_step_req)
                        for s in s_list:
                            x, y, r = s['x'], s['y'], s['r']
                            # Coordinate ROI sicure
                            x1, y1 = max(0, x - r), max(0, y - r)
                            x2, y2 = min(w_ref, x + r), min(h_ref, y + r)
                            roi = curr_img_resized[y1:y2, x1:x2]

                            is_present = check_sensor_color(roi, threshold=0.015)
                            sensor_status.append({'id': s['id'], 'step': int(step_key), 'present': is_present})

                            if is_curr_step and is_present:
                                present_cnt += 1
                else:
                    # Non allineato: restituisci stato vuoto/falso per i sensori
                    for step_key, s_list in steps_config.items():
                        for s in s_list:
                            sensor_status.append({'id': s['id'], 'step': int(step_key), 'present': False})

            # 5. UI FEEDBACK E LOGICA PUNTEGGIO
            total_targets = len(target_sensors)
            pct_sensors_found = (present_cnt / total_targets * 100) if total_targets > 0 else 0

            final_acc = 0.0
            msg = "..."
            direct = "..."

            if not is_locked:
                # Modalità Allineamento
                final_acc = alignment_score
                msg = "Allinea la forma"
                direct = "Sovrapponi la linea ROSSA al disegno"
            else:
                # Modalità Sensori (Lock attivo)
                if total_targets == 0:
                    final_acc = 100.0
                    msg = "Attesa Config"
                    direct = "..."
                elif pct_sensors_found < 100:
                    # Score parte da 60 (base lock) e sale a 99 con i sensori
                    final_acc = 60.0 + (pct_sensors_found * 0.39)
                    msg = "Cerca Sensori"
                    direct = f"Trovati {present_cnt} su {total_targets}. Tieni fermo."
                else:
                    final_acc = 100.0
                    msg = "Perfetto"
                    direct = "Step Completato!"

            return {
                'success': True,
                'accuracy': round(final_acc, 1),
                'is_locked': is_locked,  # Flag per il frontend
                'live_contour': live_contour_points,  # Coordinate per disegnare la linea rossa
                'message': msg,
                'direction': direct,
                'sensors_status': sensor_status
            }

        except Exception as e:
            logger.error(f'Validation Error: {e}')
            return {'success': False, 'error': str(e)}

    def _calculate_alignment_and_contour(self, live_img, ref_mask) -> Tuple[float, List]:
        """
        Trova il contorno principale nell'immagine live, calcola quanto sovrappone
        il riferimento, e restituisce i punti semplificati per il disegno in JS.
        """
        # 1. Preprocessing per trovare i bordi
        gray = cv2.cvtColor(live_img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        # Canny edge detection
        edges = cv2.Canny(blurred, 30, 100)

        # 2. Trova contorni
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        live_points = []
        score = 0.0

        if contours:
            # Assumiamo che il contorno più grande sia la gamba/oggetto
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)

            # Filtro anti-rumore: ignora oggetti troppo piccoli
            if area > 1000:
                # Semplifica il contorno (ApproxPolyDP) per ridurre i dati inviati al frontend
                epsilon = 0.005 * cv2.arcLength(largest_contour, True)
                approx = cv2.approxPolyDP(largest_contour, epsilon, True)

                # Converti in lista [[x,y], [x,y]] per JSON
                live_points = approx.reshape(-1, 2).tolist()

                # --- CALCOLO SCORE DI SOVRAPPOSIZIONE ---
                # Crea una maschera piena dell'oggetto live
                live_mask_filled = np.zeros_like(ref_mask)
                cv2.drawContours(live_mask_filled, [largest_contour], -1, 255, thickness=cv2.FILLED)

                # Il riferimento (ref_mask) è solitamente solo i bordi neri su sfondo bianco o viceversa.
                # Assicuriamoci di avere l'area "piena" del riferimento se possibile,
                # oppure usiamo la dilatazione dei bordi del riferimento per vedere se il live ci cade dentro.

                # Approccio robusto: Confronto bordi con tolleranza
                ref_edges = cv2.Canny(ref_mask, 50, 150)
                ref_fat = cv2.dilate(ref_edges, np.ones((15, 15), np.uint8))  # Zona valida larga

                # Quanti pixel del bordo live cadono nella zona valida del riferimento?
                # Creiamo immagine solo bordi live
                live_edges_mask = np.zeros_like(ref_mask)
                cv2.drawContours(live_edges_mask, [largest_contour], -1, 255, thickness=2)

                intersection = cv2.bitwise_and(live_edges_mask, live_edges_mask, mask=ref_fat)
                matched_pixels = cv2.countNonZero(intersection)
                total_live_pixels = cv2.countNonZero(live_edges_mask)

                if total_live_pixels > 0:
                    raw_score = (matched_pixels / total_live_pixels) * 100
                    score = min(100, raw_score)

        return score, live_points

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
