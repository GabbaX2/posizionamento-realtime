import cv2
import base64
import numpy as np
import os
from typing import Dict, Tuple
from skimage.metrics import structural_similarity as ssim


class ValidationService:
    def __init__(self):
        self.sessions: Dict[str, Dict] = {}

    def handle_reference_upload(self, request) -> Dict:
        """Handle reference image upload"""
        try:
            if 'file' not in request.files:
                return {'success': False, 'error': 'Nessun file fornito'}

            file = request.files['file']
            limb_type = request.form.get('limb_type', 'arm')
            session_id = request.form.get('session_id', 'default')

            if file.filename == '':
                return {'success': False, 'error': 'Nessun file selezionato'}

            print(f"[UPLOAD] Processing file: {file.filename}, limb_type: {limb_type}")

            # Leggi immagine
            file.seek(0)
            image_bytes = file.read()
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if image is None:
                return {'success': False, 'error': 'Impossibile leggere l\'immagine'}

            print(f"[UPLOAD] Image loaded. Size: {image.shape}")

            # Salva immagine
            temp_filename = f"{session_id}_{limb_type}_reference.png"
            upload_folder = 'uploads'
            os.makedirs(upload_folder, exist_ok=True)
            filepath = os.path.join(upload_folder, temp_filename)

            cv2.imwrite(filepath, image)
            print(f"[UPLOAD] Image saved to: {filepath}")

            # Store session data
            self.sessions[session_id] = {
                'limb_type': limb_type,
                'reference_image': filepath,
                'upload_time': str(np.datetime64('now'))
            }

            # Encode to base64
            _, buffer = cv2.imencode('.png', image)
            preview_base64 = base64.b64encode(buffer).decode()

            return {
                'success': True,
                'message': 'Immagine di riferimento caricata con successo',
                'session_id': session_id,
                'preview_image': preview_base64,
                'reference_image': preview_base64,
                'limb_type': limb_type
            }

        except Exception as e:
            print(f"[ERROR] Upload error: {e}")
            import traceback
            traceback.print_exc()
            return {'success': False, 'error': f'Errore durante il caricamento: {str(e)}'}

    def preprocess_for_comparison(self, image: np.ndarray, target_size: Tuple[int, int] = (640, 480)) -> np.ndarray:
        """
        Pre-elabora l'immagine per il confronto:
        - Ridimensiona a dimensione standard
        - Converte in scala di grigi
        - Applica blur per ridurre il rumore
        """
        # Ridimensiona mantenendo aspect ratio
        h, w = image.shape[:2]
        target_w, target_h = target_size

        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Crea immagine con padding per raggiungere target_size
        result = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        y_offset = (target_h - new_h) // 2
        x_offset = (target_w - new_w) // 2
        result[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

        # Converti in scala di grigi
        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

        # Applica blur gaussiano per ridurre rumore
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        return blurred

    def calculate_image_similarity(self, reference: np.ndarray, current: np.ndarray) -> Dict:
        """
        Calcola la somiglianza tra due immagini usando multiple metriche:
        - SSIM (Structural Similarity Index)
        - MSE (Mean Squared Error)
        - Correlazione template matching
        """
        try:
            # Pre-elabora entrambe le immagini
            ref_processed = self.preprocess_for_comparison(reference)
            curr_processed = self.preprocess_for_comparison(current)

            print(f"[SIMILARITY] Processed shapes - Ref: {ref_processed.shape}, Curr: {curr_processed.shape}")

            # 1. SSIM - Structural Similarity Index (0-1, 1 = identiche)
            ssim_score, ssim_map = ssim(ref_processed, curr_processed, full=True)
            ssim_percentage = ssim_score * 100

            print(f"[SIMILARITY] SSIM Score: {ssim_percentage:.2f}%")

            # 2. MSE - Mean Squared Error (0 = identiche)
            mse = np.mean((ref_processed.astype(float) - curr_processed.astype(float)) ** 2)
            # Normalizza MSE a percentuale (mse basso = alta somiglianza)
            mse_percentage = max(0, min(100, 100 - (mse / 100)))

            print(f"[SIMILARITY] MSE Score: {mse_percentage:.2f}%")

            # 3. Template Matching - Normalized Cross-Correlation
            result = cv2.matchTemplate(curr_processed, ref_processed, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)
            template_percentage = max(0, max_val * 100)

            print(f"[SIMILARITY] Template Match: {template_percentage:.2f}%")

            # Calcola accuratezza combinata (media pesata)
            # SSIM è il più affidabile per somiglianza strutturale
            combined_accuracy = (ssim_percentage * 0.6 + mse_percentage * 0.2 + template_percentage * 0.2)

            # Calcola offset dalla posizione ottimale
            h, w = ref_processed.shape
            center_x, center_y = w // 2, h // 2
            offset_x = max_loc[0] - center_x
            offset_y = max_loc[1] - center_y

            # Normalizza offset rispetto alle dimensioni dell'immagine
            offset_x_norm = offset_x / w
            offset_y_norm = offset_y / h

            return {
                'ssim': ssim_percentage,
                'mse': mse_percentage,
                'template_match': template_percentage,
                'combined_accuracy': combined_accuracy,
                'offset_x': offset_x_norm,
                'offset_y': offset_y_norm
            }

        except Exception as e:
            print(f"[ERROR] Similarity calculation error: {e}")
            import traceback
            traceback.print_exc()
            return {
                'ssim': 0,
                'mse': 0,
                'template_match': 0,
                'combined_accuracy': 0,
                'offset_x': 0,
                'offset_y': 0
            }

    def get_alignment_feedback(self, accuracy: float, offset_x: float, offset_y: float) -> Dict:
        """
        Genera feedback per l'utente basato su accuratezza e offset
        """
        threshold = 0.05  # 5% dello schermo
        direction_hints = []

        # Determina direzione orizzontale
        if abs(offset_x) > threshold:
            if offset_x > 0:
                direction_hints.append('SINISTRA')
            else:
                direction_hints.append('DESTRA')

        # Determina direzione verticale
        if abs(offset_y) > threshold:
            if offset_y > 0:
                direction_hints.append('SU')
            else:
                direction_hints.append('GIÙ')

        # Genera messaggio in base all'accuratezza
        if accuracy >= 95:
            message = '✓ PERFETTO! Allineamento ottimale'
            direction = None
        elif accuracy >= 90:
            message = '🎯 Eccellente! Quasi perfetto'
            direction = ' e '.join(direction_hints) if direction_hints else None
        elif accuracy >= 80:
            message = '👍 Molto bene! Piccoli aggiustamenti'
            direction = ' e '.join(direction_hints) if direction_hints else None
        elif accuracy >= 70:
            message = '🔄 Buono, continua ad allineare'
            direction = ' e '.join(direction_hints) if direction_hints else 'Riposiziona'
        elif accuracy >= 50:
            message = '⚠️ Allineamento in corso...'
            direction = ' e '.join(direction_hints) if direction_hints else 'Riposiziona'
        else:
            message = '❌ Riposizionare completamente'
            direction = ' e '.join(direction_hints) if direction_hints else 'Riposiziona'

        return {
            'message': message,
            'direction': direction
        }

    def validate_current_frame(self, request) -> Dict:
        """
        Valida il frame corrente confrontandolo con il riferimento
        usando confronto pixel-by-pixel (senza MediaPipe)
        """
        try:
            data = request.get_json()
            reference_image_data = data.get('reference_image', '')
            current_frame_data = data.get('current_frame', '')

            print(f"[VALIDATION] Ricevuti dati:")
            print(f"   - Reference image data length: {len(reference_image_data) if reference_image_data else 0}")
            print(f"   - Current frame data length: {len(current_frame_data) if current_frame_data else 0}")

            if not reference_image_data:
                return {
                    'success': False,
                    'error': 'Nessuna immagine di riferimento caricata'
                }

            if not current_frame_data:
                return {
                    'success': False,
                    'error': 'Nessun frame corrente fornito'
                }

            # Decode images
            reference = self._decode_base64_image(reference_image_data)
            if reference is None:
                return {'success': False, 'error': 'Errore nella decodifica dell\'immagine di riferimento'}

            current_frame = self._decode_base64_image(current_frame_data)
            if current_frame is None:
                return {'success': False, 'error': 'Errore nella decodifica del frame corrente'}

            print(f"[VALIDATION] Inizio calcolo somiglianza...")

            # Calcola somiglianza
            similarity_result = self.calculate_image_similarity(reference, current_frame)

            accuracy = similarity_result['combined_accuracy']
            offset_x = similarity_result['offset_x']
            offset_y = similarity_result['offset_y']

            print(f"[VALIDATION] Accuratezza combinata: {accuracy:.2f}%")
            print(f"[VALIDATION] Offset: x={offset_x:.3f}, y={offset_y:.3f}")

            # Genera feedback
            feedback = self.get_alignment_feedback(accuracy, offset_x, offset_y)

            response_data = {
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

            print(f"[VALIDATION] Response: {response_data['message']}")

            return response_data

        except Exception as e:
            print(f"[ERROR] Validation error: {e}")
            import traceback
            traceback.print_exc()
            return {'success': False, 'error': f'Errore durante la validazione: {str(e)}'}

    def get_validation_status(self, session_id: str) -> Dict:
        """Get current validation status for session"""
        if session_id not in self.sessions:
            return {
                'success': True,
                'session_id': session_id,
                'reference_loaded': False
            }

        session_data = self.sessions[session_id]

        return {
            'success': True,
            'session_id': session_id,
            'limb_type': session_data['limb_type'],
            'reference_loaded': True,
            'upload_time': session_data['upload_time']
        }

    def _decode_base64_image(self, base64_string):
        """Decodifica un'immagine base64 in formato OpenCV"""
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        try:
            img_data = base64.b64decode(base64_string)
            np_arr = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            return img
        except Exception as e:
            print(f"Errore nella decodifica immagine base64: {e}")
            return None