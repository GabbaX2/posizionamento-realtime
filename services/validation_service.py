import cv2
import base64
import numpy as np
import os
import logging
from typing import Dict, Tuple, Optional
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from utils.background_removal import remove_background_from_pil

logger = logging.getLogger(__name__)


class ValidationService:
    def __init__(self):
        self.sessions: Dict[str, Dict] = {}
        print("✅ ValidationService inizializzato.")

    def handle_reference_upload(self, request) -> Dict:
        try:
            if 'file' not in request.files:
                return {'success': False, 'error': 'Nessun file fornito'}

            file = request.files['file']
            session_id = request.form.get('session_id', 'default')

            if file.filename == '':
                return {'success': False, 'error': 'Nessun file selezionato'}

            logger.info(f"[UPLOAD] Processing file: {file.filename}")

            file.seek(0)
            nparr = np.frombuffer(file.read(), np.uint8)
            image_input = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)

            if image_input is None:
                return {'success': False, 'error': 'Impossibile leggere l\'immagine'}

            image_bgra = None

            # Gestione Sfondo
            if image_input.ndim == 3 and image_input.shape[2] == 4:
                logger.info("Immagine con trasparenza nativa rilevata.")
                image_bgra = image_input
            else:
                logger.info("Avvio rimozione sfondo automatica...")
                if len(image_input.shape) == 2:
                    image_input = cv2.cvtColor(image_input, cv2.COLOR_GRAY2BGR)
                img_rgb = cv2.cvtColor(image_input[:, :, :3], cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(img_rgb)

                pil_out, method_used = remove_background_from_pil(pil_img)

                if pil_out is not None:
                    image_bgra = cv2.cvtColor(np.array(pil_out), cv2.COLOR_RGBA2BGRA)
                    logger.info(f"✅ Sfondo rimosso usando: {method_used}")
                else:
                    logger.warning("⚠️ Rimozione sfondo fallita completamente. Uso originale opaco.")
                    b, g, r = cv2.split(image_input[:, :, :3])
                    alpha = np.ones(b.shape, dtype=b.dtype) * 255
                    image_bgra = cv2.merge((b, g, r, alpha))

            image_bgr_clean = image_bgra[:, :, :3].copy()

            # Detection
            annotated_image, contours_list, sensors_list, drawing_overlay_bgra = self._find_contours_and_sensors(
                image_bgra)

            sensors_count = len(sensors_list) if sensors_list is not None else 0
            contours_count = len(contours_list) if contours_list is not None else 0

            # Salvataggio
            upload_folder = 'uploads'
            os.makedirs(upload_folder, exist_ok=True)

            filepath = os.path.join(upload_folder, f"{session_id}_reference.png")
            cv2.imwrite(filepath, image_bgr_clean)

            if annotated_image is not None:
                cv2.imwrite(os.path.join(upload_folder, f"{session_id}_annotated.png"), annotated_image)

            if drawing_overlay_bgra is not None:
                cv2.imwrite(os.path.join(upload_folder, f"{session_id}_drawing_overlay.png"), drawing_overlay_bgra)

            self.sessions[session_id] = {
                'reference_image': filepath,
                'upload_time': str(np.datetime64('now')),
                'sensors_found': sensors_count,
                'contours_found': contours_count
            }

            def to_b64(img):
                _, buf = cv2.imencode('.png', img)
                return base64.b64encode(buf).decode()

            # Preparazione dati sensori per il frontend
            sensors_data_json = []
            if sensors_list is not None:
                for sensor in sensors_list:
                    sensors_data_json.append({
                        'x': int(sensor[0]),
                        'y': int(sensor[1]),
                        'r': int(sensor[2])
                    })

            return {
                'success': True,
                'message': 'Upload completato',
                'session_id': session_id,
                'reference_image': to_b64(image_bgr_clean),
                'annotated_image': to_b64(annotated_image) if annotated_image is not None else "",
                'drawing_overlay_image': to_b64(drawing_overlay_bgra) if drawing_overlay_bgra is not None else "",
                'sensors_found': sensors_count,
                'contours_found': contours_count,
                'sensors_data': sensors_data_json  # Dati inviati al frontend
            }

        except Exception as e:
            logger.error(f"[ERROR] Upload error: {e}", exc_info=True)
            return {'success': False, 'error': f'Errore upload: {str(e)}'}

    def _find_contours_and_sensors(self, image_bgra: np.ndarray) -> Tuple[
        np.ndarray, Optional[list], Optional[list], np.ndarray]:
        """Trova contorno principale e sensori."""
        annotated_image = image_bgra[:, :, :3].copy()
        alpha_mask = image_bgra[:, :, 3]
        height, width = image_bgra.shape[:2]

        # Rimozione Watermark
        y_end_pct = 0.08
        x_end_pct = 0.20
        y_end = int(height * y_end_pct)
        x_end = int(width * x_end_pct)

        annotated_image[0:y_end, 0:x_end] = [0, 0, 0]
        alpha_mask[0:y_end, 0:x_end] = 0

        drawing_overlay_bgra = np.zeros((height, width, 4), dtype=np.uint8)

        contour_color_bgr = (0, 255, 0)
        contour_color_bgra = (0, 255, 0, 255)
        sensor_color_bgr = (0, 0, 255)
        sensor_center_color_bgr = (0, 255, 255)
        sensor_color_bgra = (0, 0, 255, 255)
        sensor_center_color_bgra = (0, 255, 255, 255)

        # Contorni Gamba
        _, thresh = cv2.threshold(alpha_mask, 10, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            main_contour = max(contours, key=cv2.contourArea)
            cv2.drawContours(annotated_image, [main_contour], -1, contour_color_bgr, 3)
            cv2.drawContours(drawing_overlay_bgra, [main_contour], -1, contour_color_bgra, 3)

        # Sensori (Hough Circles)
        gray = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2GRAY)
        gray_blurred = cv2.GaussianBlur(gray, (9, 9), 0)

        circles = cv2.HoughCircles(
            gray_blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=70,  # <-- FIX: Aumentato per evitare sovrapposizioni
            param1=50,
            param2=26,  # <-- FIX: Aumentato per ridurre falsi positivi
            minRadius=15,
            maxRadius=45
        )

        sensors_list = []
        if circles is not None:
            circles = np.uint16(np.around(circles))
            sensors_list = circles[0, :]
            for (x, y, r) in sensors_list:
                cv2.circle(annotated_image, (x, y), r, sensor_color_bgr, 3)
                cv2.circle(annotated_image, (x, y), 2, sensor_center_color_bgr, 3)
                cv2.circle(drawing_overlay_bgra, (x, y), r, sensor_color_bgra, 3)
                cv2.circle(drawing_overlay_bgra, (x, y), 2, sensor_center_color_bgra, 3)

        return annotated_image, contours, sensors_list, drawing_overlay_bgra

    def preprocess_for_comparison(self, image: np.ndarray, target_size: Tuple[int, int] = (640, 480)) -> np.ndarray:
        h, w = image.shape[:2]
        target_w, target_h = target_size
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        result = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        y_offset = (target_h - new_h) // 2
        x_offset = (target_w - new_w) // 2
        result[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        return blurred

    def calculate_image_similarity(self, reference: np.ndarray, current: np.ndarray) -> Dict:
        """ Calcola la somiglianza applicando la maschera per ignorare lo sfondo """
        try:
            h_ref, w_ref = reference.shape[:2]
            current_resized = cv2.resize(current, (w_ref, h_ref))

            if reference.shape[2] == 4:
                mask = reference[:, :, 3]
                ref_rgb = reference[:, :, :3]
            else:
                gray_ref = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
                _, mask = cv2.threshold(gray_ref, 10, 255, cv2.THRESH_BINARY)
                ref_rgb = reference

            current_masked = cv2.bitwise_and(current_resized, current_resized, mask=mask)
            ref_masked = cv2.bitwise_and(ref_rgb, ref_rgb, mask=mask)

            ref_gray = cv2.cvtColor(ref_masked, cv2.COLOR_BGR2GRAY)
            curr_gray = cv2.cvtColor(current_masked, cv2.COLOR_BGR2GRAY)

            ref_blur = cv2.GaussianBlur(ref_gray, (5, 5), 0)
            curr_blur = cv2.GaussianBlur(curr_gray, (5, 5), 0)

            win_size = min(7, min(h_ref, w_ref))
            if win_size % 2 == 0: win_size -= 1

            ssim_score, _ = ssim(ref_blur, curr_blur, win_size=win_size, full=True)
            ssim_percentage = ssim_score * 100

            res_template = cv2.matchTemplate(current_resized, ref_rgb, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res_template)
            template_percentage = max(0, max_val * 100)

            diff = cv2.absdiff(ref_blur, curr_blur)
            non_zero_count = cv2.countNonZero(mask)

            if non_zero_count > 0:
                mse = np.sum(diff ** 2) / non_zero_count
                mse_percentage = max(0, 100 - (mse / 10))
            else:
                mse_percentage = 0

            combined_accuracy = (ssim_percentage * 0.4 + template_percentage * 0.4 + mse_percentage * 0.2)

            offset_x = (max_loc[0] - 0) / w_ref if w_ref > 0 else 0
            offset_y = (max_loc[1] - 0) / h_ref if h_ref > 0 else 0

            return {
                'ssim': ssim_percentage,
                'mse': mse_percentage,
                'template_match': template_percentage,
                'combined_accuracy': combined_accuracy,
                'offset_x': offset_x,
                'offset_y': offset_y
            }
        except Exception as e:
            logger.error(f"[ERROR] Similarity calculation error: {e}", exc_info=True)
            return {'ssim': 0, 'mse': 0, 'template_match': 0, 'combined_accuracy': 0, 'offset_x': 0, 'offset_y': 0}

    def get_alignment_feedback(self, accuracy: float, offset_x: float, offset_y: float) -> Dict:
        threshold = 0.05
        direction_hints = []
        if abs(offset_x) > threshold:
            direction_hints.append('SINISTRA' if offset_x > 0 else 'DESTRA')
        if abs(offset_y) > threshold:
            direction_hints.append('SU' if offset_y > 0 else 'GIÙ')

        if accuracy >= 95:
            message = '✓ PERFETTO! Allineamento ottimale'
        elif accuracy >= 90:
            message = '🎯 Eccellente! Quasi perfetto'
        elif accuracy >= 80:
            message = '👍 Molto bene! Piccoli aggiustamenti'
        elif accuracy >= 70:
            message = '🔄 Buono, continua ad allineare'
        elif accuracy >= 50:
            message = '⚠️ Allineamento in corso...'
        else:
            message = '❌ Riposizionare completamente'

        direction = ' e '.join(direction_hints) if direction_hints else None
        if not direction and accuracy < 95 and accuracy >= 50:
            direction = 'Centra meglio'
        elif not direction and accuracy < 50:
            direction = 'Riposiziona'

        return {'message': message, 'direction': direction}

    def validate_current_frame(self, request) -> Dict:
        try:
            data = request.get_json()
            reference_image_data = data.get('reference_image', '')
            current_frame_data = data.get('current_frame', '')

            if not reference_image_data or not current_frame_data:
                return {'success': False, 'error': 'Dati immagine mancanti'}

            reference = self._decode_base64_image(reference_image_data)
            current_frame = self._decode_base64_image(current_frame_data)

            if reference is None or current_frame is None:
                return {'success': False, 'error': 'Errore decodifica immagini'}

            similarity_result = self.calculate_image_similarity(reference, current_frame)
            accuracy = similarity_result['combined_accuracy']
            offset_x = similarity_result['offset_x']
            offset_y = similarity_result['offset_y']
            feedback = self.get_alignment_feedback(accuracy, offset_x, offset_y)

            return {
                'success': True,
                'accuracy': round(accuracy, 1),
                'message': feedback['message'],
                'direction': feedback['direction'],
                'offset_x': offset_x,
                'offset_y': offset_y,
                'metrics': {
                    'ssim': round(similarity_result['ssim'], 1),
                    'mse': round(similarity_result['mse'], 1),
                    'template_match': round(similarity_result['template_match'], 1)
                }
            }

        except Exception as e:
            logger.error(f"[ERROR] Validation error: {e}", exc_info=True)
            return {'success': False, 'error': f'Errore durante la validazione: {str(e)}'}

    def get_validation_status(self, session_id: str) -> Dict:
        if session_id not in self.sessions:
            return {'success': True, 'session_id': session_id, 'reference_loaded': False}

        session_data = self.sessions[session_id]
        return {
            'success': True,
            'session_id': session_id,
            'reference_loaded': True,
            'upload_time': session_data['upload_time'],
            'sensors_found': session_data.get('sensors_found', 0)
        }

    def _decode_base64_image(self, base64_string):
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        try:
            img_data = base64.b64decode(base64_string)
            np_arr = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            return img
        except Exception as e:
            logger.error(f"Errore nella decodifica immagine base64: {e}", exc_info=True)
            return None
