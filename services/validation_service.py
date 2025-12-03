import cv2
import base64
import numpy as np
import os
import logging
from typing import Dict, Tuple, Optional, List
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from utils.background_removal import remove_background_from_pil

logger = logging.getLogger(__name__)


class ValidationService:
    def __init__(self):
        self.sessions: Dict[str, Dict] = {}
        self.STEP_SIZE = 2
        print("✅ ValidationService: STABLE MATH MODE (40% Sim + 30% Center + 30% Sensors)")

    def handle_reference_upload(self, request) -> Dict:
        try:
            if 'file' not in request.files: return {'success': False, 'error': 'Nessun file'}
            file = request.files['file']
            session_id = request.form.get('session_id', 'default')
            if file.filename == '': return {'success': False, 'error': 'Filename vuoto'}

            file.seek(0)
            nparr = np.frombuffer(file.read(), np.uint8)
            image_input = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
            if image_input is None: return {'success': False, 'error': 'Errore decodifica'}

            image_bgra = None
            if image_input.ndim == 3 and image_input.shape[2] == 4:
                image_bgra = image_input
            else:
                if len(image_input.shape) == 2: image_input = cv2.cvtColor(image_input, cv2.COLOR_GRAY2BGR)
                img_rgb = cv2.cvtColor(image_input[:, :, :3], cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(img_rgb)
                pil_out, _ = remove_background_from_pil(pil_img)
                if pil_out is not None:
                    image_bgra = cv2.cvtColor(np.array(pil_out), cv2.COLOR_RGBA2BGRA)
                else:
                    b, g, r = cv2.split(image_input[:, :, :3])
                    alpha = np.ones(b.shape, dtype=b.dtype) * 255
                    image_bgra = cv2.merge((b, g, r, alpha))

            image_bgr_clean = image_bgra[:, :, :3].copy()

            annotated_image, contours, steps_config, drawing_overlay = self._find_contours_and_steps(image_bgra)

            total_steps = len(steps_config)
            sensors_count = sum(len(v) for v in steps_config.values())

            flat_sensors = []
            for k, v in steps_config.items(): flat_sensors.extend(v)

            upload_folder = 'uploads'
            os.makedirs(upload_folder, exist_ok=True)
            ref_path = os.path.join(upload_folder, f"{session_id}_reference.png")
            cv2.imwrite(ref_path, image_bgr_clean)
            if annotated_image is not None:
                cv2.imwrite(os.path.join(upload_folder, f"{session_id}_annotated.png"), annotated_image)
            if drawing_overlay is not None:
                cv2.imwrite(os.path.join(upload_folder, f"{session_id}_drawing_overlay.png"), drawing_overlay)

            self.sessions[session_id] = {
                'reference_image': ref_path,
                'upload_time': str(np.datetime64('now')),
                'total_steps': total_steps,
                'steps_config': steps_config
            }

            def to_b64(img):
                _, buf = cv2.imencode('.png', img)
                return base64.b64encode(buf).decode()

            return {
                'success': True,
                'message': f'OK: {sensors_count} sensori.',
                'session_id': session_id,
                'reference_image': to_b64(image_bgr_clean),
                'annotated_image': to_b64(annotated_image) if annotated_image is not None else "",
                'drawing_overlay_image': to_b64(drawing_overlay) if drawing_overlay is not None else "",
                'total_steps': total_steps,
                'steps_config': steps_config,
                'sensors_data': flat_sensors
            }

        except Exception as e:
            logger.error(f"Upload error: {e}", exc_info=True)
            return {'success': False, 'error': str(e)}

    def _find_contours_and_steps(self, image_bgra: np.ndarray) -> Tuple[np.ndarray, list, Dict[str, List], np.ndarray]:
        annotated_image = image_bgra[:, :, :3].copy()
        alpha_mask = image_bgra[:, :, 3]
        h, w = image_bgra.shape[:2]

        y_end, x_end = int(h * 0.08), int(w * 0.20)
        annotated_image[0:y_end, 0:x_end] = 0
        alpha_mask[0:y_end, 0:x_end] = 0

        drawing_overlay = np.zeros((h, w, 4), dtype=np.uint8)

        _, thresh = cv2.threshold(alpha_mask, 10, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            main_cnt = max(contours, key=cv2.contourArea)
            cv2.drawContours(annotated_image, [main_cnt], -1, (0, 255, 0), 2)
            cv2.drawContours(drawing_overlay, [main_cnt], -1, (0, 255, 0, 255), 2)

        gray = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2GRAY)
        gray_filtered = cv2.medianBlur(gray, 5)

        circles = cv2.HoughCircles(
            gray_filtered, cv2.HOUGH_GRADIENT, dp=1.2,
            minDist=70, param1=50, param2=35, minRadius=22, maxRadius=65
        )

        valid_sensors = []
        if circles is not None:
            circles = np.uint16(np.around(circles))
            raw_circles = circles[0, :]
            for (x, y, r) in raw_circles:
                x1, y1 = max(0, x - r), max(0, y - r)
                x2, y2 = min(w, x + r), min(h, y + r)
                roi = annotated_image[y1:y2, x1:x2]
                if roi.size == 0: continue
                mean, std_dev = cv2.meanStdDev(roi)
                contrast = np.mean(std_dev)
                if contrast > 10.0: valid_sensors.append((x, y, r))

            valid_sensors = self._remove_overlapping_circles(valid_sensors, min_dist=60)

        valid_sensors = sorted(valid_sensors, key=lambda val: val[1])
        steps_config = {}
        current_step = 1

        for i in range(0, len(valid_sensors), self.STEP_SIZE):
            chunk = valid_sensors[i: i + self.STEP_SIZE]
            step_sensors = []
            for idx_in_chunk, (x, y, r) in enumerate(chunk):
                sensor_global_id = i + idx_in_chunk + 1

                cv2.circle(annotated_image, (x, y), r, (0, 0, 255), 3)
                cv2.circle(drawing_overlay, (x, y), r, (0, 0, 255, 255), 3)
                cv2.rectangle(annotated_image, (x - 10, y - 10), (x + 15, y + 10), (0, 0, 255), -1)
                cv2.putText(annotated_image, str(sensor_global_id), (x - 5, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (255, 255, 255), 2)
                cv2.putText(drawing_overlay, str(sensor_global_id), (x - 5, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (255, 255, 255), 2)

                step_sensors.append({
                    'id': int(sensor_global_id), 'step': int(current_step),
                    'x': int(x), 'y': int(y), 'r': int(r)
                })

            steps_config[str(current_step)] = step_sensors
            current_step += 1

        return annotated_image, contours, steps_config, drawing_overlay

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

    def _verify_sensor_presence_strict(self, frame: np.ndarray, x: int, y: int, r: int) -> bool:
        """ LIVE CHECK: Permissivo """
        try:
            h, w = frame.shape[:2]
            margin = int(r * 0.9)
            x1, y1 = max(0, x - r - margin), max(0, y - r - margin)
            x2, y2 = min(w, x + r + margin), min(h, y + r + margin)
            roi = frame[y1:y2, x1:x2]
            if roi.size == 0: return False

            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            lower_blue = np.array([70, 15, 20])
            upper_blue = np.array([160, 255, 255])
            lower_white = np.array([0, 0, 40])
            upper_white = np.array([180, 90, 255])

            mask = cv2.inRange(hsv, lower_blue, upper_blue) + cv2.inRange(hsv, lower_white, upper_white)
            return (cv2.countNonZero(mask) / (roi.shape[0] * roi.shape[1])) > 0.02

        except Exception:
            return False

    def validate_current_frame(self, request) -> Dict:
        try:
            data = request.get_json()
            ref_b64 = data.get('reference_image')
            curr_b64 = data.get('current_frame')
            session_id = data.get('session_id', 'default')
            current_step_req = str(data.get('current_step', 1))

            if not ref_b64 or not curr_b64: return {'success': False, 'error': 'No data'}

            ref_img = self._decode_base64_image(ref_b64)
            curr_img = self._decode_base64_image(curr_b64)
            if ref_img is None or curr_img is None: return {'success': False, 'error': 'Img Err'}

            steps_config = self.sessions.get(session_id, {}).get('steps_config', {})

            # --- 1. Calcolo GEOMETRIA (Match Template) ---
            # Questo dice quanto l'immagine assomiglia al riferimento
            sim_res = self.calculate_image_similarity(ref_img, curr_img)
            similarity_score = sim_res['combined_accuracy']  # 0-100

            # --- 2. Calcolo CENTRATURA (Offset) ---
            # Se la gamba è al bordo, il punteggio deve scendere
            centering_score = self.calculate_centering_score(sim_res['offset_x'], sim_res['offset_y'])

            # --- 3. Calcolo SENSORI ---
            sensors_status = []
            target_sensors = steps_config.get(current_step_req, [])
            present_cnt = 0

            scale_x = curr_img.shape[1] / ref_img.shape[1]
            scale_y = curr_img.shape[0] / ref_img.shape[0]

            for step_key, s_list in steps_config.items():
                is_curr = (str(step_key) == current_step_req)
                for s in s_list:
                    sx, sy, sr = int(s['x'] * scale_x), int(s['y'] * scale_y), int(s['r'] * scale_x)

                    is_present = self._verify_sensor_presence_strict(curr_img, sx, sy, sr)

                    sensors_status.append({'id': s['id'], 'step': int(step_key), 'present': is_present})
                    if is_curr and is_present: present_cnt += 1

            total_targets = len(target_sensors)
            sensors_score = (present_cnt / total_targets * 100.0) if total_targets > 0 else 100.0

            # --- FORMULA DI STABILITA' ---
            # 40% Somiglianza Immagine (È la gamba giusta?)
            # 30% Centratura (È nel posto giusto?)
            # 30% Sensori (Ci sono i cerchi?)

            final_acc = (similarity_score * 0.40) + (centering_score * 0.30) + (sensors_score * 0.30)

            # Cap di sicurezza: Se non vedo la gamba (similarity < 50), il voto è 0.
            if similarity_score < 50.0:
                final_acc = 0.0

            msg = ""
            direct = ""
            if final_acc >= 85:
                msg = "✓ Eccellente"
                direct = None
            elif final_acc >= 60:
                if centering_score < 80:
                    msg = "Centra meglio"
                    direct = "Sposta la gamba"
                elif sensors_score < 100:
                    msg = "Sensori mancanti"
                    direct = "Controlla elettrodi"
                else:
                    msg = "Allinea meglio"
                    direct = "Piccoli movimenti"
            else:
                msg = "Riposiziona"
                direct = "Inquadra la gamba"

            return {
                'success': True, 'accuracy': round(final_acc, 1),
                'message': msg, 'direction': direct,
                'sensors_status': sensors_status,
                'metrics': sim_res
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def calculate_centering_score(self, off_x, off_y):
        """ Restituisce 100 se offset è 0, scende a 0 se offset è grande """
        # Offset è normalizzato (0-1). 0.1 di offset è già tanto.
        dist = np.sqrt(off_x ** 2 + off_y ** 2)
        # Tolleranza: fino a 0.05 (5%) è perfetto (100 punti)
        # Da 0.05 a 0.30 scende linearmente a 0
        if dist < 0.05: return 100.0
        if dist > 0.30: return 0.0
        return 100.0 - ((dist - 0.05) / 0.25 * 100.0)

    def calculate_image_similarity(self, reference: np.ndarray, current: np.ndarray) -> Dict:
        try:
            h, w = reference.shape[:2]
            curr_res = cv2.resize(current, (w, h))

            # Use simple grayscale conversion
            ref_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
            curr_gray = cv2.cvtColor(curr_res, cv2.COLOR_BGR2GRAY)

            # Standardize lighting
            ref_gray = cv2.normalize(ref_gray, None, 0, 255, cv2.NORM_MINMAX)
            curr_gray = cv2.normalize(curr_gray, None, 0, 255, cv2.NORM_MINMAX)

            # Robust Template Match (Normed handles lighting differences best)
            res = cv2.matchTemplate(curr_gray, ref_gray, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)

            # max_val is -1 to 1. Convert to 0-100.
            # Usually good match is > 0.6
            match_score = max(0, max_val * 100)

            # Calculate Offset from center (0,0 because we resized 1:1)
            offset_x = max_loc[0] / w
            offset_y = max_loc[1] / h

            return {
                'combined_accuracy': match_score,  # Just raw matching score
                'offset_x': offset_x, 'offset_y': offset_y,
                'ssim': 0, 'template_match': match_score
            }
        except:
            return {'combined_accuracy': 0, 'offset_x': 0, 'offset_y': 0}

    def get_alignment_feedback(self, acc, ox, oy):
        if acc >= 85: return {'message': '✓ OK', 'direction': None}
        return {'message': 'Allinea', 'direction': 'Sposta'}

    def _decode_base64_image(self, b64):
        if ',' in b64: b64 = b64.split(',')[1]
        try:
            return cv2.imdecode(np.frombuffer(base64.b64decode(b64), np.uint8), cv2.IMREAD_COLOR)
        except:
            return None
