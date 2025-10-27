import cv2
import base64
import numpy as np
import os
from typing import Dict, List

from models.limb_detector import LimbDetector
from utils.image_processing import save_uploaded_file, encode_image_to_base64
from utils.background_removal import remove_background, remove_background_from_file


class ValidationService:
    def __init__(self):
        self.sessions: Dict[str, Dict] = {}

    def handle_reference_upload(self, request) -> Dict:
        """Handle reference image upload with background removal"""
        try:
            # Validate request
            if 'file' not in request.files:
                return {
                    'success': False,
                    'error': 'Nessun file fornito'
                }

            file = request.files['file']
            limb_type = request.form.get('limb_type', 'arm')
            session_id = request.form.get('session_id', 'default')

            if file.filename == '':
                return {'success': False, 'error': 'Nessun file selezionato'}

            print(f"[UPLOAD] Processing file: {file.filename}, limb_type: {limb_type}, session: {session_id}")

            # RIMUOVI LO SFONDO PRIMA DI SALVARE
            print("[UPLOAD] Rimozione sfondo in corso...")
            file.seek(0)
            image_no_bg_base64 = remove_background_from_file(file)

            if image_no_bg_base64 is None:
                print("[UPLOAD] Errore nella rimozione dello sfondo, uso immagine originale")
                file.seek(0)
                image_bytes = file.read()
                nparr = np.frombuffer(image_bytes, np.uint8)
                test_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            else:
                print("[UPLOAD] Sfondo rimosso con successo")
                image_bytes = base64.b64decode(image_no_bg_base64)
                nparr = np.frombuffer(image_bytes, np.uint8)
                test_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if test_image is None:
                return {'success': False, 'error': 'Impossibile leggere l\'immagine. Formato non supportato.'}

            print(f"[UPLOAD] Image loaded successfully. Size: {test_image.shape}")

            # Test limb detection
            limb_detector = LimbDetector(limb_type)
            detected_points = limb_detector.get_limb_points(test_image)

            if detected_points is None:
                return {
                    'success': False,
                    'error': f'Arto non rilevato nell\'immagine. Assicurati che il {limb_type} sia completamente visibile e ben illuminato.'
                }

            print(f"[UPLOAD] Limb detected. Points: {len(detected_points) if detected_points else 0}")

            # Save processed image
            temp_filename = f"{session_id}_{limb_type}_reference.png"
            upload_folder = 'uploads'
            os.makedirs(upload_folder, exist_ok=True)
            filepath = os.path.join(upload_folder, temp_filename)

            cv2.imwrite(filepath, test_image)
            print(f"[UPLOAD] Processed image saved to: {filepath}")

            # Store session data
            self.sessions[session_id] = {
                'limb_type': limb_type,
                'reference_image': filepath,
                'upload_time': str(np.datetime64('now')),
                'reference_points': detected_points
            }

            # Encode image to base64 for preview
            _, buffer = cv2.imencode('.png', test_image)
            preview_base64 = base64.b64encode(buffer).decode()

            response = {
                'success': True,
                'message': 'Immagine di riferimento caricata con successo',
                'session_id': session_id,
                'preview_image': preview_base64,
                'reference_image': preview_base64,  # Immagine senza sfondo per overlay
                'points_detected': len(detected_points),
                'limb_type': limb_type
            }

            print(f"[UPLOAD] Success! Response: {response}")
            return response

        except Exception as e:
            print(f"[ERROR] Unexpected error in handle_reference_upload: {e}")
            import traceback
            traceback.print_exc()
            return {'success': False, 'error': f'Errore durante il caricamento: {str(e)}'}

    def validate_current_frame(self, request) -> Dict:
        """Validate current frame - ora restituisce solo dati per overlay"""
        try:
            data = request.get_json()
            reference_image_data = data.get('reference_image', '')
            limb_type = None

            print(f"[VALIDATION] Ricevuti dati:")
            print(f"   - Reference image data length: {len(reference_image_data) if reference_image_data else 0}")

            metadata_str = data.get('reference_metadata')
            print(f"   - Metadata string: {metadata_str}")

            if metadata_str:
                try:
                    import json
                    meta = json.loads(metadata_str)
                    limb_type = meta.get('limbType')
                    print(f"   - Limb type from metadata: {limb_type}")
                except Exception as e:
                    print(f"   - Error parsing metadata: {e}")
                    limb_type = None

            if not reference_image_data:
                return {'success': False,
                        'error': 'Nessuna immagine di riferimento caricata. Vai alla pagina "Carica Riferimento".'}

            # Decode reference image
            reference = self._decode_base64_image(reference_image_data)
            if reference is None:
                return {'success': False, 'error': 'Errore nella decodifica dell\'immagine di riferimento'}

            limb_type = limb_type or 'arm'

            # Rileva punti anatomici sul riferimento (per verificare che sia valido)
            print(f"[VALIDATION] Verifica punti anatomici sul riferimento...")
            limb_detector = LimbDetector(limb_type)
            reference_points = limb_detector.get_limb_points(reference)

            print(f"[VALIDATION] Punti anatomici rilevati: {len(reference_points) if reference_points else 0}")

            if reference_points is None:
                return {'success': False, 'error': 'Impossibile rilevare punti di riferimento sull\'immagine caricata.'}

            # Restituisci l'immagine di riferimento per l'overlay
            _, buffer = cv2.imencode('.png', reference)
            reference_base64 = base64.b64encode(buffer).decode()

            response_data = {
                'success': True,
                'reference_image': reference_base64,
                'limb_type': limb_type,
                'points_detected': len(reference_points),
                'message': 'Immagine di riferimento pronta per overlay'
            }

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

        status = {
            'success': True,
            'session_id': session_id,
            'limb_type': session_data['limb_type'],
            'reference_loaded': True,
            'upload_time': session_data['upload_time'],
            'reference_points': len(session_data.get('reference_points', []))
        }

        return status

    def _decode_base64_image(self, base64_string):
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

    def annotate_reference_image(self, request) -> Dict:
        """
        Annota un'immagine di riferimento con rimozione sfondo
        Endpoint: /api/annotate_reference
        """
        try:
            data = request.get_json()
            image_data = data.get('image', '')
            limb_type = data.get('limb_type', 'arm')

            print(f"[ANNOTATE] Ricevuta richiesta per annotazione, limb_type: {limb_type}")

            if not image_data:
                return {'success': False, 'error': 'Nessun dato immagine fornito'}

            # RIMUOVI LO SFONDO
            print("[ANNOTATE] Rimozione sfondo in corso...")
            image_no_bg = remove_background(image_data)

            if image_no_bg is None:
                print("[ANNOTATE] Fallback: uso immagine originale")
                image_to_process = image_data
            else:
                print("[ANNOTATE] Sfondo rimosso con successo")
                image_to_process = f"data:image/png;base64,{image_no_bg}"

            # Decodifica l'immagine
            if ',' in image_to_process:
                image_to_process = image_to_process.split(',')[1]

            image_bytes = base64.b64decode(image_to_process)
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if image is None:
                return {
                    'success': False,
                    'error': 'Impossibile decodificare l\'immagine'
                }

            print(f"[ANNOTATE] Immagine decodificata con successo. Size: {image.shape}")

            # Rileva punti anatomici
            limb_detector = LimbDetector(limb_type)
            print("[ANNOTATE] Rilevamento punti anatomici...")
            detected_points = limb_detector.get_limb_points(image)

            if detected_points is None:
                return {
                    'success': False,
                    'error': 'Nessun arto rilevato nell\'immagine'
                }

            print(f"[ANNOTATE] Punti anatomici rilevati: {len(detected_points) if detected_points else 0}")

            # Converti in base64 (immagine senza sfondo)
            _, buffer = cv2.imencode('.png', image)
            annotated_base64 = base64.b64encode(buffer).decode()

            return {
                'success': True,
                'annotated_image': annotated_base64,
                'points_detected': len(detected_points) if detected_points else 0,
                'limb_type': limb_type
            }

        except Exception as e:
            print(f"[ERROR] annotate_reference_image: {e}")
            import traceback
            traceback.print_exc()
            return {
                'success': False,
                'error': str(e)
            }