import cv2
import base64
import numpy as np
import os
from typing import Dict, Tuple, Optional
from skimage.metrics import structural_similarity as ssim

# Import del detector SENIAM
try:
    from models.limb_detector import SENIAMLimbDetector

    DETECTOR_AVAILABLE = True
except ImportError:
    print("⚠️ SENIAMLimbDetector non disponibile")
    DETECTOR_AVAILABLE = False


class ValidationService:
    def __init__(self):
        self.sessions: Dict[str, Dict] = {}
        self.detector = None

        # Inizializza il detector se disponibile
        if DETECTOR_AVAILABLE:
            try:
                self.detector = SENIAMLimbDetector(
                    limb_type='leg',
                    model_name='lower_limb',
                    target_input_size=(512, 512)
                )
                print("✅ SENIAMLimbDetector inizializzato")
            except Exception as e:
                print(f"⚠️ Errore inizializzazione detector: {e}")
                self.detector = None

    def handle_reference_upload(self, request) -> Dict:
        """Handle reference image upload with landmark detection"""
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

            # Processa con landmark detection se disponibile e tipo è 'leg'
            landmarks_data = None
            annotated_image = None

            if limb_type == 'leg' and self.detector is not None:
                print("[DETECTION] Inizio rilevamento landmark SENIAM...")
                try:
                    landmarks_data = self._detect_and_annotate(image)

                    if landmarks_data and landmarks_data.get('landmarks'):
                        # CORREZIONE: Passa anche sensor_positions
                        annotated_image = self._draw_landmarks_on_image(
                            image.copy(),
                            landmarks_data['landmarks'],
                            landmarks_data.get('sensor_positions', {})  # ← AGGIUNTO
                        )
                        print(f"[DETECTION] ✅ Rilevati {len(landmarks_data['landmarks'])} landmark")
                        print(
                            f"[DETECTION] ✅ Calcolate {len(landmarks_data.get('sensor_positions', {}))} posizioni sensori")
                    else:
                        print("[DETECTION] ⚠️ Nessun landmark rilevato")

                except Exception as e:
                    print(f"[DETECTION] ❌ Errore detection: {e}")
                    import traceback
                    traceback.print_exc()

            # Salva immagine originale
            temp_filename = f"{session_id}_{limb_type}_reference.png"
            upload_folder = 'uploads'
            os.makedirs(upload_folder, exist_ok=True)
            filepath = os.path.join(upload_folder, temp_filename)
            cv2.imwrite(filepath, image)
            print(f"[UPLOAD] Image saved to: {filepath}")

            # Salva anche immagine annotata se disponibile
            annotated_filepath = None
            if annotated_image is not None:
                annotated_filename = f"{session_id}_{limb_type}_annotated.png"
                annotated_filepath = os.path.join(upload_folder, annotated_filename)
                cv2.imwrite(annotated_filepath, annotated_image)
                print(f"[UPLOAD] Annotated image saved to: {annotated_filepath}")

            # Store session data
            session_data = {
                'limb_type': limb_type,
                'reference_image': filepath,
                'upload_time': str(np.datetime64('now'))
            }

            if landmarks_data:
                session_data['landmarks'] = landmarks_data
                session_data['annotated_image'] = annotated_filepath

            self.sessions[session_id] = session_data

            # Encode images to base64
            _, buffer = cv2.imencode('.png', image)
            preview_base64 = base64.b64encode(buffer).decode()

            annotated_base64 = None
            if annotated_image is not None:
                _, ann_buffer = cv2.imencode('.png', annotated_image)
                annotated_base64 = base64.b64encode(ann_buffer).decode()

            response = {
                'success': True,
                'message': 'Immagine di riferimento caricata con successo',
                'session_id': session_id,
                'preview_image': preview_base64,
                'reference_image': preview_base64,
                'limb_type': limb_type
            }

            # Aggiungi dati landmark se disponibili
            if landmarks_data:
                response['landmarks'] = landmarks_data
                if annotated_base64:
                    response['annotated_image'] = annotated_base64

            return response

        except Exception as e:
            print(f"[ERROR] Upload error: {e}")
            import traceback
            traceback.print_exc()
            return {'success': False, 'error': f'Errore durante il caricamento: {str(e)}'}

    def _detect_and_annotate(self, image: np.ndarray) -> Optional[Dict]:
        """Rileva landmark usando SENIAMLimbDetector"""
        if self.detector is None:
            return None

        try:
            # Rileva landmark
            landmarks = self.detector.detect_landmarks(
                image,
                score_threshold=0.2
            )

            if not landmarks:
                return None

            # Conta landmark validi
            valid_landmarks = {
                name: info for name, info in landmarks.items()
                if info.get('xy') is not None
            }

            if not valid_landmarks:
                return None

            # Calcola posizioni sensori per muscoli principali
            sensor_positions = {}
            muscles_to_detect = ['vasto_mediale', 'retto_femorale', 'gastrocnemio', 'tibiale_anteriore']

            for muscle in muscles_to_detect:
                try:
                    sensor_pos = self.detector.calculate_sensor_position(
                        landmarks,
                        muscle,
                        min_score=0.2
                    )
                    sensor_positions[muscle] = sensor_pos
                    print(f"   ✓ Sensore {muscle}: posizione {sensor_pos['position']}")
                except Exception as e:
                    print(f"   ⚠️ Impossibile calcolare posizione per {muscle}: {e}")
                    continue

            return {
                'landmarks': landmarks,
                'sensor_positions': sensor_positions,
                'total_landmarks': len(valid_landmarks),
                'detection_method': 'SENIAMLimbDetector'
            }

        except Exception as e:
            print(f"[ERROR] Errore in _detect_and_annotate: {e}")
            return None

    def _draw_landmarks_on_image(self, image: np.ndarray, landmarks: Dict,
                                 sensor_positions: Dict = None) -> np.ndarray:
        """
        Disegna i landmark e le posizioni dei sensori sull'immagine

        Args:
            image: immagine da annotare
            landmarks: Dict con formato {'name': {'xy': (x,y), 'score': float}}
            sensor_positions: Dict con formato {'muscle_name': {'position': (x,y), 'confidence': float, ...}}
        """
        result = image.copy()
        h, w = image.shape[:2]

        # 1. DISEGNA I LANDMARK ANATOMICI
        print(f"[DRAW] Disegno {len(landmarks)} landmark anatomici...")
        for name, info in landmarks.items():
            xy = info.get('xy')
            score = info.get('score', 0.0)

            if xy is None:
                print(f"   ⚠️ Landmark '{name}' non ha coordinate valide")
                continue

            x, y = xy
            print(f"   ✓ Disegno landmark '{name}' a ({x}, {y}) con score {score:.2f}")

            # Colore basato sulla confidenza
            if score >= 0.75:
                color = (0, 255, 0)  # Verde - alta confidenza
                label_color = "HIGH"
            elif score >= 0.4:
                color = (0, 255, 255)  # Giallo - media confidenza
                label_color = "MED"
            else:
                color = (0, 0, 255)  # Rosso - bassa confidenza
                label_color = "LOW"

            # Disegna cerchio per il landmark
            cv2.circle(result, (x, y), 8, color, -1)  # Cerchio pieno
            cv2.circle(result, (x, y), 10, (255, 255, 255), 2)  # Bordo bianco

            # Prepara etichetta
            label = f"{name}"
            label_score = f"{score:.2f}"

            # Background per il testo
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 2
            (text_width, text_height), baseline = cv2.getTextSize(
                label, font, font_scale, thickness
            )

            # Posizione testo (sopra il punto)
            text_x = x + 12
            text_y = y - 10

            # Rettangolo background
            cv2.rectangle(
                result,
                (text_x - 2, text_y - text_height - 2),
                (text_x + text_width + 2, text_y + baseline + 2),
                (0, 0, 0),
                -1
            )

            # Testo landmark name
            cv2.putText(
                result, label, (text_x, text_y),
                font, font_scale, color, thickness
            )

            # Score sotto il nome
            cv2.putText(
                result, label_score, (text_x, text_y + 15),
                font, 0.4, (255, 255, 255), 1
            )

        # 2. DISEGNA LE POSIZIONI DEI SENSORI
        if sensor_positions:
            print(f"[DRAW] Disegno {len(sensor_positions)} posizioni sensori...")

            # Colori diversi per muscoli diversi
            sensor_colors = {
                'vasto_mediale': (255, 0, 0),  # Blu
                'retto_femorale': (255, 0, 255),  # Magenta
                'gastrocnemio': (0, 165, 255),  # Arancione
                'tibiale_anteriore': (0, 255, 0)  # Verde
            }

            for muscle_name, sensor_info in sensor_positions.items():
                position = sensor_info.get('position')
                confidence = sensor_info.get('confidence', 0.0)

                if position is None:
                    print(f"   ⚠️ Sensore '{muscle_name}' non ha posizione valida")
                    continue

                x, y = position
                color = sensor_colors.get(muscle_name, (128, 128, 128))

                print(f"   ✓ Disegno sensore '{muscle_name}' a ({x}, {y}) con confidenza {confidence:.2f}")

                # Disegna cerchio grande per il sensore
                cv2.circle(result, (x, y), 15, color, -1)  # Cerchio pieno colorato
                cv2.circle(result, (x, y), 18, (255, 255, 255), 3)  # Bordo bianco spesso
                cv2.circle(result, (x, y), 21, (0, 0, 0), 2)  # Bordo nero esterno

                # Croce centrale per precisione
                cross_size = 8
                cv2.line(result, (x - cross_size, y), (x + cross_size, y), (255, 255, 255), 2)
                cv2.line(result, (x, y - cross_size), (x, y + cross_size), (255, 255, 255), 2)

                # Etichetta sensore
                label = f"SENSORE: {muscle_name}"
                conf_label = f"Conf: {confidence:.2f}"

                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6
                thickness = 2

                # Calcola dimensioni testo
                (text_width, text_height), baseline = cv2.getTextSize(
                    label, font, font_scale, thickness
                )

                # Posizione testo (a destra del sensore)
                text_x = x + 25
                text_y = y

                # Background nero per leggibilità
                cv2.rectangle(
                    result,
                    (text_x - 3, text_y - text_height - 3),
                    (text_x + text_width + 3, text_y + 20),
                    (0, 0, 0),
                    -1
                )

                # Bordo colorato
                cv2.rectangle(
                    result,
                    (text_x - 3, text_y - text_height - 3),
                    (text_x + text_width + 3, text_y + 20),
                    color,
                    2
                )

                # Testo nome muscolo (bianco)
                cv2.putText(
                    result, label, (text_x, text_y),
                    font, font_scale, (255, 255, 255), thickness
                )

                # Testo confidenza (colore del sensore)
                cv2.putText(
                    result, conf_label, (text_x, text_y + 15),
                    font, 0.45, color, 1
                )

                # Linea che collega i landmark usati al sensore
                if 'landmarks_used' in sensor_info:
                    landmarks_used = sensor_info['landmarks_used']
                    if len(landmarks_used) >= 2:
                        lm_start_name = landmarks_used[0]
                        lm_end_name = landmarks_used[1]

                        lm_start = landmarks.get(lm_start_name, {}).get('xy')
                        lm_end = landmarks.get(lm_end_name, {}).get('xy')

                        if lm_start and lm_end:
                            # Linea tratteggiata tra i landmark
                            self._draw_dashed_line(result, lm_start, lm_end, color, 2)

        print(f"[DRAW] ✅ Annotazione completata")
        return result

    def _draw_dashed_line(self, image: np.ndarray, pt1: Tuple[int, int],
                          pt2: Tuple[int, int], color: Tuple[int, int, int],
                          thickness: int = 1, dash_length: int = 10):
        """Disegna una linea tratteggiata tra due punti"""
        x1, y1 = pt1
        x2, y2 = pt2

        dist = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        dashes = int(dist / dash_length)

        for i in range(dashes):
            if i % 2 == 0:  # Disegna solo i segmenti pari
                start = (
                    int(x1 + (x2 - x1) * i / dashes),
                    int(y1 + (y2 - y1) * i / dashes)
                )
                end = (
                    int(x1 + (x2 - x1) * (i + 1) / dashes),
                    int(y1 + (y2 - y1) * (i + 1) / dashes)
                )
                cv2.line(image, start, end, color, thickness, cv2.LINE_AA)

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

        response = {
            'success': True,
            'session_id': session_id,
            'limb_type': session_data['limb_type'],
            'reference_loaded': True,
            'upload_time': session_data['upload_time']
        }

        # Aggiungi dati landmark se presenti
        if 'landmarks' in session_data:
            response['landmarks'] = session_data['landmarks']

        return response

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
