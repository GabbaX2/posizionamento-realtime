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
        print("✅ ValidationService: SMART DETECTION (Shape + Color Verification)")

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

            # Gestione RGBA/Background
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

            # --- RILEVAMENTO INTELLIGENTE ---
            annotated_image, contours, steps_config, drawing_overlay = self._find_contours_and_steps(image_bgra)

            total_steps = len(steps_config)
            sensors_count = sum(len(v) for v in steps_config.values())
            flat_sensors = []
            for k, v in steps_config.items(): flat_sensors.extend(v)

            # Salvataggio immagini debug
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
                'message': f'Rilevati {sensors_count} sensori validi.',
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

        # Maschera per ignorare angoli (rumore UI)
        y_end, x_end = int(h * 0.08), int(w * 0.20)
        annotated_image[0:y_end, 0:x_end] = 0
        alpha_mask[0:y_end, 0:x_end] = 0

        drawing_overlay = np.zeros((h, w, 4), dtype=np.uint8)

        # Trova contorno gamba
        _, thresh = cv2.threshold(alpha_mask, 10, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            main_cnt = max(contours, key=cv2.contourArea)
            cv2.drawContours(annotated_image, [main_cnt], -1, (0, 255, 0), 2)
            cv2.drawContours(drawing_overlay, [main_cnt], -1, (0, 255, 0, 255), 2)

        # Rilevamento Cerchi
        gray = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2GRAY)
        gray_filtered = cv2.medianBlur(gray, 5)

        # Param2 alzato leggermente (30 -> 35) per ridurre falsi positivi geometrici
        circles = cv2.HoughCircles(
            gray_filtered, cv2.HOUGH_GRADIENT, dp=1.2,
            minDist=60, param1=50, param2=35, minRadius=15, maxRadius=55
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

                # --- FIX CRUCIALE: FILTRO COLORE ---
                # Non basta che sia rotondo. Deve avere i colori del sensore.
                if self._check_color_signature(roi):
                    valid_sensors.append((x, y, r))
                else:
                    # Debug visivo: Disegna in Giallo i cerchi scartati (opzionale)
                    # cv2.circle(annotated_image, (x, y), r, (0, 255, 255), 1)
                    pass

            valid_sensors = self._remove_overlapping_circles(valid_sensors, min_dist=50)

        # Ordinamento e creazione Step
        valid_sensors = sorted(valid_sensors, key=lambda val: val[1])
        steps_config = {}
        current_step = 1

        for i in range(0, len(valid_sensors), self.STEP_SIZE):
            chunk = valid_sensors[i: i + self.STEP_SIZE]
            step_sensors = []
            for idx_in_chunk, (x, y, r) in enumerate(chunk):
                sensor_global_id = i + idx_in_chunk + 1

                # Disegno finale (Cerchi rossi confermati)
                cv2.circle(annotated_image, (x, y), r, (0, 0, 255), 3)
                cv2.circle(drawing_overlay, (x, y), r, (0, 0, 255, 255), 3)

                cv2.rectangle(annotated_image, (x - 10, y - 10), (x + 15, y + 10), (0, 0, 255), -1)
                cv2.putText(annotated_image, str(sensor_global_id), (x - 5, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (255, 255, 255), 2)
                cv2.putText(drawing_overlay, str(sensor_global_id), (x - 5, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (255, 255, 255), 2)

                step_sensors.append(
                    {'id': int(sensor_global_id), 'step': int(current_step), 'x': int(x), 'y': int(y), 'r': int(r)})

            steps_config[str(current_step)] = step_sensors
            current_step += 1

        return annotated_image, contours, steps_config, drawing_overlay

    def _check_color_signature(self, roi: np.ndarray) -> bool:
        """
        Controlla se la ROI contiene i colori specifici del sensore (Azzurro/Grigio/Bianco).
        Serve a scartare falsi positivi sulla pelle.
        """
        try:
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

            # 1. AZZURRO (Gel)
            lower_cyan = np.array([75, 20, 70])
            upper_cyan = np.array([145, 255, 255])
            mask_cyan = cv2.inRange(hsv, lower_cyan, upper_cyan)

            # 2. GRIGIO (Bottone)
            lower_grey = np.array([0, 0, 50])
            upper_grey = np.array([180, 50, 220])  # Sat max bassa per evitare pelle arrossata
            mask_grey = cv2.inRange(hsv, lower_grey, upper_grey)

            # 3. BIANCO (Bordo)
            lower_white = np.array([0, 0, 160])
            upper_white = np.array([180, 60, 255])
            mask_white = cv2.inRange(hsv, lower_white, upper_white)

            combined_mask = cv2.bitwise_or(mask_cyan, mask_grey)
            combined_mask = cv2.bitwise_or(combined_mask, mask_white)

            non_zero = cv2.countNonZero(combined_mask)
            total_pixels = roi.shape[0] * roi.shape[1]

            # Deve avere almeno il 5% di pixel "Sensor-Like" per essere accettato
            # La pelle normale avrà quasi 0% su queste maschere specifiche.
            return (non_zero / total_pixels) > 0.05

        except Exception:
            return False

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

    # --- VALIDAZIONE LIVE (User Experience Mode) ---
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

            # 1. GEOMETRIA (SSIM Boosted)
            sim_res = self.calculate_image_similarity(ref_img, curr_img)
            raw_ssim = sim_res['raw_ssim']

            if raw_ssim >= 50:
                alignment_score = 90 + ((raw_ssim - 50) * 0.2)
            elif raw_ssim >= 30:
                alignment_score = 75 + ((raw_ssim - 30) / 20 * 15)
            else:
                alignment_score = raw_ssim * 2

            # 2. SENSORI
            sensors_status = []
            target_sensors = steps_config.get(current_step_req, [])
            present_cnt = 0
            scale_x = curr_img.shape[1] / ref_img.shape[1]
            scale_y = curr_img.shape[0] / ref_img.shape[0]

            for step_key, s_list in steps_config.items():
                is_curr_step = (str(step_key) == current_step_req)
                for s in s_list:
                    sx, sy, sr = int(s['x'] * scale_x), int(s['y'] * scale_y), int(s['r'] * scale_x)
                    # Usiamo la stessa logica colore per la validazione
                    is_present = self._check_sensor_presence_live(curr_img, sx, sy, sr)
                    sensors_status.append({'id': s['id'], 'step': int(step_key), 'present': is_present})
                    if is_curr_step and is_present: present_cnt += 1

            total_targets = len(target_sensors)
            sensors_score = (present_cnt / total_targets * 100.0) if total_targets > 0 else 100.0

            # CALCOLO FINALE
            if alignment_score >= 75:
                final_acc = (alignment_score * 0.40) + (sensors_score * 0.60)
                if sensors_score >= 99.0: final_acc = max(final_acc, 88.0)
            else:
                final_acc = alignment_score

            final_acc = min(100.0, final_acc)

            if final_acc >= 85:
                msg, direct = "✓ Eccellente", None
            elif alignment_score < 75:
                msg, direct = "Riposiziona", "Inquadra la gamba"
            elif sensors_score < 100:
                msg, direct = "Sensori?", "Controlla elettrodi"
            else:
                msg, direct = "Stabile...", "Non muoverti"

            return {
                'success': True, 'accuracy': round(final_acc, 1),
                'message': msg, 'direction': direct,
                'sensors_status': sensors_status,
                'metrics': {'raw_ssim': round(raw_ssim, 1), 'boosted': round(alignment_score, 1)}
            }
        except Exception as e:
            logger.error(f"Validation error: {e}")
            return {'success': False, 'error': str(e)}

    def _check_sensor_presence_live(self, frame, x, y, r):
        """ Wrapper per validazione live, usa area leggermente ridotta """
        h, w = frame.shape[:2]
        check_r = int(r * 0.8)  # Area più stretta per il live
        x1, y1 = max(0, x - check_r), max(0, y - check_r)
        x2, y2 = min(w, x + check_r), min(h, y + check_r)
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0: return False

        # Tolleranza più bassa per il LIVE (1.5%) vs Rilevamento iniziale (5%)
        # Perché in live ci sono riflessi e movimento
        return self._check_color_signature_percent(roi, threshold=0.015)

    def _check_color_signature_percent(self, roi, threshold):
        """ Logica core dei colori riutilizzabile """
        try:
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            # Azzurro
            lower_cyan = np.array([75, 20, 70])
            upper_cyan = np.array([145, 255, 255])
            mask_cyan = cv2.inRange(hsv, lower_cyan, upper_cyan)
            # Grigio
            lower_grey = np.array([0, 0, 50])
            upper_grey = np.array([180, 60, 220])
            mask_grey = cv2.inRange(hsv, lower_grey, upper_grey)
            # Bianco
            lower_white = np.array([0, 0, 150])
            upper_white = np.array([180, 70, 255])
            mask_white = cv2.inRange(hsv, lower_white, upper_white)

            combined = cv2.bitwise_or(mask_cyan, mask_grey)
            combined = cv2.bitwise_or(combined, mask_white)
            ratio = cv2.countNonZero(combined) / (roi.shape[0] * roi.shape[1])
            return ratio > threshold
        except:
            return False

    def calculate_image_similarity(self, reference: np.ndarray, current: np.ndarray) -> Dict:
        try:
            h, w = reference.shape[:2]
            curr_resized = cv2.resize(current, (w, h))
            grayA = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
            grayB = cv2.cvtColor(curr_resized, cv2.COLOR_BGR2GRAY)
            score, _ = ssim(grayA, grayB, full=True, win_size=7, channel_axis=None)
            raw_ssim = max(0, score * 100)
            return {'raw_ssim': raw_ssim}
        except:
            return {'raw_ssim': 0}

    def _decode_base64_image(self, b64):
        if not b64: return None
        if ',' in b64: b64 = b64.split(',')[1]
        try:
            return cv2.imdecode(np.frombuffer(base64.b64decode(b64), np.uint8), cv2.IMREAD_COLOR)
        except:
            return None
