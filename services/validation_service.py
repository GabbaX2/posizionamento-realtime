import cv2
import base64
import numpy as np
import os
from typing import Dict, Tuple, Optional
from skimage.metrics import structural_similarity as ssim


# NOTA: Non importiamo più nulla da 'models.limb_detector'

class ValidationService:
    def __init__(self):
        """
        Inizializza il servizio. Non c'è più nessun detector da caricare.
        """
        self.sessions: Dict[str, Dict] = {}
        print("✅ ValidationService (modalità Contorni/Sensori) inizializzato.")

    def handle_reference_upload(self, request) -> Dict:
        """
        Gestisce l'upload dell'immagine di riferimento.
        Ora rileva contorni e sensori (cerchi).
        """
        try:
            if 'file' not in request.files:
                return {'success': False, 'error': 'Nessun file fornito'}

            file = request.files['file']
            session_id = request.form.get('session_id', 'default')

            if file.filename == '':
                return {'success': False, 'error': 'Nessun file selezionato'}

            print(f"[UPLOAD] Processing file: {file.filename} per sessione {session_id}")

            # Leggi immagine
            file.seek(0)
            image_bytes = file.read()
            nparr = np.frombuffer(image_bytes, np.uint8)

            # --- MODIFICA CHIAVE ---
            # Carichiamo l'immagine "UNCHANGED" per preservare il canale Alpha (trasparenza)
            image_bgra = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)

            if image_bgra is None:
                return {'success': False, 'error': 'Impossibile leggere l\'immagine'}

            # Verifica che l'immagine abbia 4 canali (BGR + Alpha)
            if image_bgra.ndim < 3 or image_bgra.shape[2] != 4:
                print(f"❌ Errore: l'immagine non ha un canale alpha. Shape: {image_bgra.shape}")
                return {
                    'success': False,
                    'error': "Immagine non ha sfondo trasparente. Assicurati che 'Rimuovi Sfondo' sia attivo e abbia funzionato."
                }

            print(f"[UPLOAD] Immagine con sfondo rimosso caricata. Shape: {image_bgra.shape}")

            # Estrai la parte BGR per il confronto e l'annotazione
            image_bgr_clean = image_bgra[:, :, :3].copy()

            # --- NUOVA LOGICA: Trova Contorni e Sensori ---
            print("[DETECTION] Inizio rilevamento contorni e sensori...")
            # Modifica: _find_contours_and_sensors ora restituisce 4 valori
            annotated_image, contours_list, sensors_list, drawing_overlay_bgra = self._find_contours_and_sensors(
                image_bgra)

            sensors_count = len(sensors_list) if sensors_list is not None else 0
            contours_count = len(contours_list) if contours_list is not None else 0

            print(f"[DETECTION] ✅ Rilevati {contours_count} contorni e {sensors_count} sensori (cerchi).")

            # Salva l'immagine BGR PULITA (senza disegni) per il confronto SSIM
            temp_filename = f"{session_id}_reference.png"
            upload_folder = 'uploads'
            os.makedirs(upload_folder, exist_ok=True)
            filepath = os.path.join(upload_folder, temp_filename)
            cv2.imwrite(filepath, image_bgr_clean)
            print(f"[UPLOAD] Immagine di riferimento (pulita) salvata in: {filepath}")

            # Salva l'immagine ANNOTATA (con disegni)
            annotated_filepath = None
            if annotated_image is not None:
                annotated_filename = f"{session_id}_annotated.png"
                annotated_filepath = os.path.join(upload_folder, annotated_filename)
                cv2.imwrite(annotated_filepath, annotated_image)
                print(f"[UPLOAD] Immagine annotata salvata in: {annotated_filepath}")

            # --- NUOVA PARTE: Salva l'immagine "Solo Disegno" (con trasparenza) ---
            drawing_overlay_filepath = None
            if drawing_overlay_bgra is not None:
                drawing_overlay_filename = f"{session_id}_drawing_overlay.png"
                drawing_overlay_filepath = os.path.join(upload_folder, drawing_overlay_filename)
                # Salviamo il file BGRA (con trasparenza)
                cv2.imwrite(drawing_overlay_filepath, drawing_overlay_bgra)
                print(f"[UPLOAD] Immagine 'Solo Disegno' salvata in: {drawing_overlay_filepath}")

            # Salva i dati della sessione
            session_data = {
                'reference_image': filepath,
                'annotated_image': annotated_filepath,
                'drawing_overlay_image': drawing_overlay_filepath,  # AGGIUNTO
                'upload_time': str(np.datetime64('now')),
                'sensors_found': sensors_count,
                'contours_found': contours_count
            }
            self.sessions[session_id] = session_data

            # Codifica in base64 per l'anteprima (immagine pulita)
            _, buffer_preview = cv2.imencode('.png', image_bgr_clean)
            preview_base64 = base64.b64encode(buffer_preview).decode()

            # Codifica in base64 per l'immagine ANNOTATA (sopra la gamba)
            _, buffer_annotated = cv2.imencode('.png', annotated_image)
            annotated_base64 = base64.b64encode(buffer_annotated).decode()

            # --- NUOVA PARTE: Codifica in base64 per l'immagine Solo Disegno ---
            _, buffer_drawing_overlay = cv2.imencode('.png', drawing_overlay_bgra)
            drawing_overlay_base64 = base64.b64encode(buffer_drawing_overlay).decode()

            response = {
                'success': True,
                'message': 'Immagine di riferimento caricata e analizzata con successo',
                'session_id': session_id,
                'reference_image': preview_base64,  # Immagine pulita (per SSIM)
                'annotated_image': annotated_base64,  # Immagine con disegni sopra la gamba
                'drawing_overlay_image': drawing_overlay_base64,  # <<< NUOVO: Immagine Solo Disegno
                'sensors_found': sensors_count,
                'contours_found': contours_count  # Aggiunto per i risultati
            }

            return response

        except Exception as e:
            print(f"[ERROR] Upload error: {e}")
            import traceback
            traceback.print_exc()
            return {'success': False, 'error': f'Errore durante il caricamento: {str(e)}'}

    def _find_contours_and_sensors(self, image_bgra: np.ndarray) -> Tuple[
        np.ndarray, Optional[list], Optional[list], np.ndarray]:
        """
        Trova il contorno principale e i sensori (cerchi) su un'immagine BGRA.

        Args:
            image_bgra: Immagine a 4 canali (BGR+Alpha)

        Returns:
            Tuple: (immagine_annotata_bgr, lista_contorni, lista_cerchi, immagine_solo_disegno_bgra)
        """
        # 1. Crea l'immagine di output e ottieni la maschera
        annotated_image = image_bgra[:, :, :3].copy()
        alpha_mask = image_bgra[:, :, 3]

        # --- CREAZIONE NUOVO OUTPUT (SOLO DISEGNO) ---
        # Crea un'immagine BGRA completamente trasparente (altezza, larghezza, 4 canali)
        height, width = image_bgra.shape[:2]
        # Inizializzata a zeri: BGR=0, Alpha=0 (trasparente)
        drawing_overlay_bgra = np.zeros((height, width, 4), dtype=np.uint8)

        # Definisci colori con Alpha opaco (255)
        contour_color_bgr = (0, 255, 0)  # Verde
        contour_color_bgra = (0, 255, 0, 255)  # Verde opaco (Alpha=255)
        sensor_color_bgr = (0, 0, 255)  # Rosso
        sensor_center_color_bgr = (0, 255, 255)  # Giallo
        sensor_color_bgra = (0, 0, 255, 255)  # Rosso opaco
        sensor_center_color_bgra = (0, 255, 255, 255)  # Giallo opaco

        # 2. Trova il contorno principale (la gamba)
        # Usiamo la maschera alpha per trovare i contorni
        _, thresh = cv2.threshold(alpha_mask, 10, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        main_contour = None
        if contours:
            # Trova il contorno più grande (che dovrebbe essere la gamba)
            main_contour = max(contours, key=cv2.contourArea)

            # Disegna il contorno sulla nostra immagine di output (sopra la gamba)
            cv2.drawContours(annotated_image, [main_contour], -1, contour_color_bgr, 3)

            # Disegna il contorno sulla nuova immagine trasparente
            cv2.drawContours(drawing_overlay_bgra, [main_contour], -1, contour_color_bgra, 3)

        # 3. Trova i sensori (cerchi)
        # Converti l'immagine BGR in scala di grigi
        gray = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2GRAY)

        # Applica un blur per ridurre il rumore e aiutare il rilevamento dei cerchi
        gray_blurred = cv2.medianBlur(gray, 5)

        # Usa la Trasformata di Hough per trovare i cerchi
        circles = cv2.HoughCircles(
            gray_blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=30,  # Distanza minima tra i centri dei cerchi
            param1=50,  # Soglia alta per Canny edge detector
            param2=25,  # <<< MODIFICA CHIAVE: Ridotto da 35 a 25 per maggiore sensibilità
            minRadius=10,  # Raggio minimo (pixel)
            maxRadius=60  # Raggio massimo (pixel)
        )

        sensors_list = []
        if circles is not None:
            circles = np.uint16(np.around(circles))
            sensors_list = circles[0, :]

            # Disegna i cerchi trovati
            for (x, y, r) in sensors_list:
                # Disegna su immagine annotata (sopra la gamba)
                cv2.circle(annotated_image, (x, y), r, sensor_color_bgr, 3)  # Rosso
                cv2.circle(annotated_image, (x, y), 2, sensor_center_color_bgr, 3)  # Giallo

                # Disegna su immagine trasparente
                cv2.circle(drawing_overlay_bgra, (x, y), r, sensor_color_bgra, 3)
                cv2.circle(drawing_overlay_bgra, (x, y), 2, sensor_center_color_bgra, 3)

        return annotated_image, contours, sensors_list, drawing_overlay_bgra

    # ... (Metodi rimanenti)
    def preprocess_for_comparison(self, image: np.ndarray, target_size: Tuple[int, int] = (640, 480)) -> np.ndarray:
        """ Pre-elabora l'immagine per il confronto SSIM """
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
        """ Calcola la somiglianza tra due immagini """
        try:
            ref_processed = self.preprocess_for_comparison(reference)
            curr_processed = self.preprocess_for_comparison(current)

            ssim_score, ssim_map = ssim(ref_processed, curr_processed, full=True)
            ssim_percentage = ssim_score * 100

            mse = np.mean((ref_processed.astype(float) - curr_processed.astype(float)) ** 2)
            mse_percentage = max(0, min(100, 100 - (mse / 100)))

            if ref_processed.shape[0] > curr_processed.shape[0] or ref_processed.shape[1] > curr_processed.shape[1]:
                template_percentage = 0.0
                max_loc = (0, 0)
            else:
                result = cv2.matchTemplate(curr_processed, ref_processed, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(result)
                template_percentage = max(0, max_val * 100)

            combined_accuracy = (ssim_percentage * 0.6 + mse_percentage * 0.2 + template_percentage * 0.2)

            h, w = ref_processed.shape
            center_x = 0
            center_y = 0
            offset_x = max_loc[0] - center_x
            offset_y = max_loc[1] - center_y
            offset_x_norm = offset_x / w if w > 0 else 0
            offset_y_norm = offset_y / h if h > 0 else 0

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
            return {'ssim': 0, 'mse': 0, 'template_match': 0, 'combined_accuracy': 0, 'offset_x': 0, 'offset_y': 0}

    def get_alignment_feedback(self, accuracy: float, offset_x: float, offset_y: float) -> Dict:
        """ Genera feedback per l'utente """
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
        """ Valida il frame corrente confrontandolo con il riferimento """
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
            return response_data

        except Exception as e:
            print(f"[ERROR] Validation error: {e}")
            return {'success': False, 'error': f'Errore durante la validazione: {str(e)}'}

    def get_validation_status(self, session_id: str) -> Dict:
        """Ottiene lo stato della validazione per la sessione"""
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
            'reference_loaded': True,
            'upload_time': session_data['upload_time'],
            'sensors_found': session_data.get('sensors_found', 0)
            # Abbiamo rimosso limb_type e view_type perché non più usati
        }

        # Non ci sono più 'landmarks' da aggiungere
        return response

    def _decode_base64_image(self, base64_string):
        """Decodifica un'immagine base64 in formato OpenCV"""
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        try:
            img_data = base64.b64decode(base64_string)
            np_arr = np.frombuffer(img_data, np.uint8)
            # Usiamo IMREAD_COLOR qui, assumendo che i frame attuali non siano BGRA
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            return img
        except Exception as e:
            print(f"Errore nella decodifica immagine base64: {e}")
            return None
