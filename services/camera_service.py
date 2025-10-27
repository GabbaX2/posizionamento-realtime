import cv2
import threading
import time
import platform
from typing import Optional


class CameraService:
    def __init__(self):
        self.camera = None
        self.is_running = False
        self.current_frame = None
        self.lock = threading.Lock()
        self.camera_thread = None

    def _try_open(self, source, api_preference=None):
        try:
            if api_preference is not None:
                cap = cv2.VideoCapture(source, api_preference)
            else:
                cap = cv2.VideoCapture(source)
            if cap is None or not cap.isOpened():
                if cap:
                    cap.release()
                return None
            return cap
        except Exception:
            return None

    def start_camera(self, camera_id: int = 0) -> bool:
        """Start camera capture with macOS fallbacks"""
        try:
            system = platform.system()
            # List of attempts: (source, api)
            attempts = []

            # Default generic attempt
            attempts.append((camera_id, None))

            if system == 'Darwin':
                # macOS: try AVFoundation backend if available
                # OpenCV constants may differ; try using cv2.CAP_AVFOUNDATION if defined
                try:
                    attempts.append((camera_id, cv2.CAP_AVFOUNDATION))
                except Exception:
                    pass
                # try VideoCapture with string source (avfoundation:<index>)
                attempts.append((f"avfoundation:{camera_id}", cv2.CAP_FFMPEG))
                # try with GStreamer fallback (if built)
                try:
                    attempts.append((camera_id, cv2.CAP_GSTREAMER))
                except Exception:
                    pass

            else:
                # Windows/Linux common backends
                try:
                    attempts.append((camera_id, cv2.CAP_DSHOW))
                except Exception:
                    pass
                try:
                    attempts.append((camera_id, cv2.CAP_V4L2))
                except Exception:
                    pass

            cap = None
            for src, api in attempts:
                cap = self._try_open(src, api)
                if cap:
                    print(f"Camera aperta con source={src} api={api}")
                    break

            if not cap:
                print("Nessuna camera trovata con i backend provati")
                return False

            self.camera = cap
            self.is_running = True
            self.camera_thread = threading.Thread(target=self._capture_frames)
            self.camera_thread.daemon = True
            self.camera_thread.start()
            return True
        except Exception as e:
            print(f'Error starting camera: {e}')
            return False

    def stop_camera(self) -> None:
        """Stop camera capture"""
        self.is_running = False
        if self.camera_thread and self.camera_thread.is_alive():
            self.camera_thread.join(timeout=2.0)

        if self.camera:
            self.camera.release()
            self.camera = None

    def get_current_frame(self) -> Optional[bytes]:
        """Get current frame as JPEG bytes"""
        with self.lock:
            if self.current_frame is None:
                return None

            ret, buffer = cv2.imencode('.jpg', self.current_frame)
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