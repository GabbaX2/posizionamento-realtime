import threading
import time
import platform
import cv2
from typing import Optional, Tuple


class CameraService:
    def __init__(self):
        self.camera = None
        self.is_running = False
        self.current_frame = None
        self.lock = threading.Lock()
        self.camera_thread = None
        self.target_resolution = (1280, 720)  # Default resolution

    def _try_open(self, source, api_preference=None, width=None, height=None):
        """Tenta di aprire la camera con parametri specifici"""
        try:
            if api_preference is not None:
                cap = cv2.VideoCapture(source, api_preference)
            else:
                cap = cv2.VideoCapture(source)

            if cap is None or not cap.isOpened():
                if cap:
                    cap.release()
                return None

            # Imposta risoluzione se specificata
            if width and height:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

                # Verifica risoluzione effettiva ottenuta
                actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                print(f"Risoluzione richiesta: {width}x{height}, ottenuta: {actual_width}x{actual_height}")

            return cap
        except Exception as e:
            print(f"Errore nell'apertura camera: {e}")
            return None

    def start_camera(self, camera_id: int = 0, width: int = None, height: int = None) -> bool:
        """
        Start camera capture con risoluzione specificata

        Args:
            camera_id: ID della camera (default 0)
            width: Larghezza desiderata (None = auto)
            height: Altezza desiderata (None = auto)
        """
        try:
            # Salva la risoluzione target
            if width and height:
                self.target_resolution = (width, height)

            system = platform.system()
            attempts = []

            # Tentativo con risoluzione specifica
            attempts.append((camera_id, None, width, height))

            if system == 'Darwin':
                # macOS: prova AVFoundation
                try:
                    attempts.append((camera_id, cv2.CAP_AVFOUNDATION, width, height))
                except Exception:
                    pass
                attempts.append((f"avfoundation:{camera_id}", cv2.CAP_FFMPEG, width, height))
                try:
                    attempts.append((camera_id, cv2.CAP_GSTREAMER, width, height))
                except Exception:
                    pass
            else:
                # Windows/Linux
                try:
                    attempts.append((camera_id, cv2.CAP_DSHOW, width, height))
                except Exception:
                    pass
                try:
                    attempts.append((camera_id, cv2.CAP_V4L2, width, height))
                except Exception:
                    pass

            # Fallback senza risoluzione specifica
            attempts.append((camera_id, None, None, None))

            cap = None
            for src, api, w, h in attempts:
                cap = self._try_open(src, api, w, h)
                if cap:
                    print(f"✅ Camera aperta: source={src}, api={api}, resolution={w}x{h}")
                    break

            if not cap:
                print("❌ Nessuna camera trovata")
                return False

            self.camera = cap
            self.is_running = True
            self.camera_thread = threading.Thread(target=self._capture_frames)
            self.camera_thread.daemon = True
            self.camera_thread.start()
            return True

        except Exception as e:
            print(f'❌ Errore avvio camera: {e}')
            return False

    def update_resolution(self, width: int, height: int) -> bool:
        """
        Aggiorna la risoluzione della camera mantenendo lo stream attivo

        Args:
            width: Nuova larghezza
            height: Nuova altezza

        Returns:
            True se l'aggiornamento è riuscito
        """
        if not self.camera or not self.is_running:
            print("Camera non attiva, impossibile aggiornare risoluzione")
            return False

        try:
            with self.lock:
                self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

                actual_width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))

                self.target_resolution = (actual_width, actual_height)
                print(f"✅ Risoluzione aggiornata: {actual_width}x{actual_height}")
                return True

        except Exception as e:
            print(f"❌ Errore aggiornamento risoluzione: {e}")
            return False

    def get_current_resolution(self) -> Optional[Tuple[int, int]]:
        """Ottiene la risoluzione corrente della camera"""
        if not self.camera:
            return None

        try:
            width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
            return (width, height)
        except Exception:
            return None

    def stop_camera(self) -> None:
        """Stop camera capture"""
        self.is_running = False
        if self.camera_thread and self.camera_thread.is_alive():
            self.camera_thread.join(timeout=2.0)

        if self.camera:
            self.camera.release()
            self.camera = None

    def get_current_frame(self, target_size: Tuple[int, int] = None) -> Optional[bytes]:
        """
        Get current frame as JPEG bytes

        Args:
            target_size: Tuple (width, height) per ridimensionare il frame.
                        Se None, usa la risoluzione originale.

        Returns:
            Frame in formato JPEG bytes o None
        """
        with self.lock:
            if self.current_frame is None:
                return None

            frame = self.current_frame.copy()

        # Ridimensiona se richiesto
        if target_size:
            try:
                frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)
            except Exception as e:
                print(f"Errore ridimensionamento frame: {e}")

        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if ret:
            return buffer.tobytes()

        return None

    def _capture_frames(self) -> None:
        """Capture frames in separate thread"""
        while self.is_running and self.camera:
            ret, frame = self.camera.read()
            if ret:
                with self.lock:
                    self.current_frame = frame
            else:
                time.sleep(0.1)
