import cv2
import mediapipe as mp
import numpy as np
from typing import Dict, Optional, Tuple
import os
import ssl
import urllib.request

# DISABILITA VERIFICA SSL PRIMA DI TUTTO
ssl._create_default_https_context = ssl._create_unverified_context

# Imposta variabile d'ambiente per MediaPipe
os.environ['MEDIAPIPE_DISABLE_SSL'] = '1'


class LimbDetector:
    def __init__(self, limb_type='arm', model_complexity=1):
        """
        Inizializza il rilevatore di arti

        Args:
            limb_type: 'arm' o 'leg'
            model_complexity: 0, 1, o 2 (0 è più veloce, 2 più accurato)
        """
        self.limb_type = limb_type
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = None
        self.is_dummy = False

        print(f"🔧 Inizializzazione LimbDetector per '{limb_type}'...")

        # Inizializza MediaPipe con gestione errori migliorata
        self._initialize_pose(model_complexity)

        # Definisce le connessioni per ogni tipo di arto
        self.limb_connections = {
            'arm': [
                (self.mp_pose.PoseLandmark.LEFT_SHOULDER, self.mp_pose.PoseLandmark.LEFT_ELBOW),
                (self.mp_pose.PoseLandmark.LEFT_ELBOW, self.mp_pose.PoseLandmark.LEFT_WRIST)
            ],
            'leg': [
                (self.mp_pose.PoseLandmark.LEFT_HIP, self.mp_pose.PoseLandmark.LEFT_KNEE),
                (self.mp_pose.PoseLandmark.LEFT_KNEE, self.mp_pose.PoseLandmark.LEFT_ANKLE)
            ]
        }

        # Indici dei landmark per facilitare l'accesso
        self.landmark_indices = {
            'arm': {
                'shoulder': self.mp_pose.PoseLandmark.LEFT_SHOULDER,
                'elbow': self.mp_pose.PoseLandmark.LEFT_ELBOW,
                'wrist': self.mp_pose.PoseLandmark.LEFT_WRIST
            },
            'leg': {
                'hip': self.mp_pose.PoseLandmark.LEFT_HIP,
                'knee': self.mp_pose.PoseLandmark.LEFT_KNEE,
                'ankle': self.mp_pose.PoseLandmark.LEFT_ANKLE
            }
        }

    def _initialize_pose(self, model_complexity):
        """Inizializza MediaPipe Pose con fallback progressivi"""
        configs = [
            # Primo tentativo: configurazione ottimale
            {
                'static_image_mode': True,
                'model_complexity': model_complexity,
                'smooth_landmarks': True,
                'enable_segmentation': False,
                'min_detection_confidence': 0.5,
                'min_tracking_confidence': 0.5
            },
            # Secondo tentativo: configurazione semplificata
            {
                'static_image_mode': True,
                'model_complexity': 1,
                'smooth_landmarks': False,
                'enable_segmentation': False,
                'min_detection_confidence': 0.3,
                'min_tracking_confidence': 0.3
            },
            # Terzo tentativo: configurazione minimale
            {
                'static_image_mode': True,
                'model_complexity': 0,
                'min_detection_confidence': 0.3
            }
        ]

        for i, config in enumerate(configs):
            try:
                print(f"   Tentativo {i + 1}/{len(configs)} con config: {config}")
                self.pose = self.mp_pose.Pose(**config)
                print(f"✅ MediaPipe inizializzato per {self.limb_type} (config {i + 1})")
                self.is_dummy = False
                return
            except Exception as e:
                print(f"⚠️ Tentativo {i + 1} fallito: {type(e).__name__}: {e}")
                if i == len(configs) - 1:
                    print("❌ Impossibile inizializzare MediaPipe dopo tutti i tentativi")
                    print("🔧 ATTENZIONE: Creando detector fittizio per testing...")
                    self._create_dummy_detector()

    def _create_dummy_detector(self):
        """Crea un detector fittizio per testing quando MediaPipe fallisce"""
        self.is_dummy = True

        class DummyPose:
            def __init__(self, mp_pose, limb_type):
                self.mp_pose = mp_pose
                self.limb_type = limb_type

            def process(self, image):
                h, w = image.shape[:2]

                class DummyResult:
                    def __init__(self, mp_pose, limb_type, width, height):
                        self._mp_pose = mp_pose
                        self._limb_type = limb_type
                        self._width = width
                        self._height = height
                        self._landmarks = None

                    @property
                    def pose_landmarks(self):
                        if self._landmarks is None:
                            self._landmarks = self._create_landmarks()
                        return self._landmarks

                    def _create_landmarks(self):
                        class DummyLandmarks:
                            def __init__(self, mp_pose, limb_type, width, height):
                                self._mp_pose = mp_pose
                                self._limb_type = limb_type
                                self._width = width
                                self._height = height

                            @property
                            def landmark(self):
                                class DummyLandmark:
                                    def __init__(self, x, y, visibility=0.9):
                                        self.x = x
                                        self.y = y
                                        self.visibility = visibility

                                # Crea 33 landmark (standard MediaPipe)
                                landmarks = [DummyLandmark(0.5, 0.5, 0.0)] * 33

                                # Imposta landmark specifici basati sul tipo di arto
                                if self._limb_type == 'arm':
                                    # Posizioni relative per un braccio sinistro
                                    landmarks[self._mp_pose.PoseLandmark.LEFT_SHOULDER] = DummyLandmark(0.35, 0.25,
                                                                                                        0.95)
                                    landmarks[self._mp_pose.PoseLandmark.LEFT_ELBOW] = DummyLandmark(0.30, 0.45, 0.95)
                                    landmarks[self._mp_pose.PoseLandmark.LEFT_WRIST] = DummyLandmark(0.25, 0.65, 0.95)
                                else:  # leg
                                    landmarks[self._mp_pose.PoseLandmark.LEFT_HIP] = DummyLandmark(0.40, 0.35, 0.95)
                                    landmarks[self._mp_pose.PoseLandmark.LEFT_KNEE] = DummyLandmark(0.38, 0.60, 0.95)
                                    landmarks[self._mp_pose.PoseLandmark.LEFT_ANKLE] = DummyLandmark(0.36, 0.85, 0.95)

                                return landmarks

                        return DummyLandmarks(self._mp_pose, self._limb_type, self._width, self._height)

                return DummyResult(self.mp_pose, self.limb_type, w, h)

        self.pose = DummyPose(self.mp_pose, self.limb_type)
        print("✅ Detector fittizio creato per testing")
        print("⚠️  ATTENZIONE: I risultati potrebbero non essere accurati!")

    def preprocess_image(self, image):
        """Preprocessa l'immagine per migliorare il rilevamento"""
        if image is None or image.size == 0:
            raise ValueError("Immagine non valida")

        h, w = image.shape[:2]
        print(f"   📏 Dimensioni immagine originale: {w}x{h}")

        target_size = 640

        # Ridimensiona se necessario
        if max(h, w) > target_size * 2 or max(h, w) < target_size / 2:
            scale = target_size / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            print(f"   🔧 Immagine ridimensionata a: {new_w}x{new_h}")

        # Migliora contrasto con CLAHE
        try:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            enhanced = cv2.merge([l, a, b])
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
            print(f"   ✅ Contrasto migliorato con CLAHE")
            return enhanced
        except Exception as e:
            print(f"   ⚠️ Errore nel miglioramento contrasto: {e}, uso immagine originale")
            return image

    def get_limb_points(self, image):
        """
        Estrae i punti chiave dell'arto dall'immagine

        Returns:
            Dict con le coordinate dei punti chiave o None se il rilevamento fallisce
        """
        print(f"\n🔍 Rilevamento punti per '{self.limb_type}'...")

        if self.pose is None:
            print("❌ MediaPipe Pose non inizializzato")
            return None

        if self.is_dummy:
            print("⚠️  Usando detector fittizio - risultati non reali!")

        try:
            # Verifica immagine
            if image is None or image.size == 0:
                print("❌ Immagine non valida")
                return None

            print(f"   📸 Immagine ricevuta: shape={image.shape}, dtype={image.dtype}")

            # Preprocessa l'immagine
            processed_image = self.preprocess_image(image)

            # Converti in RGB (MediaPipe richiede RGB)
            rgb_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
            print(f"   ✅ Immagine convertita in RGB")

            # Processa l'immagine
            print(f"   🔄 Processing con MediaPipe...")
            results = self.pose.process(rgb_image)

            if not results.pose_landmarks:
                print("❌ Nessun landmark rilevato nell'immagine")
                print("💡 Suggerimenti:")
                print("   - Assicurati che l'arto sia ben visibile")
                print("   - Migliora l'illuminazione")
                print("   - Evita sfondi troppo complessi")
                print("   - Verifica che la persona sia di fronte alla camera")
                return None

            print(f"   ✅ Landmark rilevati!")

            # Estrai i punti dell'arto specifico
            landmarks = results.pose_landmarks.landmark
            h, w = image.shape[:2]

            points = {}
            landmark_map = self.landmark_indices.get(self.limb_type, {})

            for point_name, landmark_idx in landmark_map.items():
                landmark = landmarks[landmark_idx]

                # Verifica la visibilità del landmark
                visibility = landmark.visibility if hasattr(landmark, 'visibility') else 1.0

                if visibility < 0.3:
                    print(f"   ⚠️ Landmark '{point_name}' poco visibile (visibility: {visibility:.2f})")
                else:
                    print(f"   ✓ Landmark '{point_name}': visibility={visibility:.2f}")

                points[point_name] = self._landmark_to_pixel(landmark, w, h)

            print(f"✅ {len(points)} punti '{self.limb_type}' rilevati: {list(points.keys())}")

            # Stampa le coordinate
            for name, (x, y) in points.items():
                print(f"   • {name}: ({x}, {y})")

            return points

        except Exception as e:
            print(f'❌ Errore in get_limb_points: {type(e).__name__}: {e}')
            import traceback
            traceback.print_exc()
            return None

    def _landmark_to_pixel(self, landmark, image_width: int, image_height: int) -> Tuple[int, int]:
        """Converte coordinate normalizzate in pixel"""
        x = int(np.clip(landmark.x * image_width, 0, image_width - 1))
        y = int(np.clip(landmark.y * image_height, 0, image_height - 1))
        return (x, y)

    def draw_limb_on_image(self, image: np.ndarray, points: Dict) -> np.ndarray:
        """Disegna l'arto rilevato sull'immagine"""
        if points is None or len(points) == 0:
            print("⚠️ Nessun punto da disegnare")
            return image

        result_image = image.copy()

        # Disegna le connessioni tra i punti
        point_names = list(self.landmark_indices[self.limb_type].keys())
        for i in range(len(point_names) - 1):
            start_name = point_names[i]
            end_name = point_names[i + 1]

            if start_name in points and end_name in points:
                cv2.line(result_image, points[start_name], points[end_name],
                         (0, 255, 0), 3, cv2.LINE_AA)

        # Disegna i punti chiave
        for point_name, (x, y) in points.items():
            # Cerchio esterno
            cv2.circle(result_image, (x, y), 8, (0, 255, 0), 2)
            # Cerchio interno
            cv2.circle(result_image, (x, y), 4, (0, 0, 255), -1)
            # Etichetta
            cv2.putText(result_image, point_name, (x + 12, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(result_image, point_name, (x + 12, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)

        return result_image

    def get_limb_orientation(self, points: Dict) -> float:
        """
        Calcola l'orientamento dell'arto in gradi

        Returns:
            Angolo in gradi (-180 a 180)
        """
        if points is None or len(points) < 2:
            return 0.0

        try:
            if self.limb_type == 'arm':
                start_point = points.get('shoulder')
                end_point = points.get('wrist')
            else:  # leg
                start_point = points.get('hip')
                end_point = points.get('ankle')

            if start_point is None or end_point is None:
                return 0.0

            dx = end_point[0] - start_point[0]
            dy = end_point[1] - start_point[1]
            angle = np.degrees(np.arctan2(dy, dx))

            return angle
        except Exception as e:
            print(f"⚠️ Errore nel calcolo dell'orientamento: {e}")
            return 0.0

    def get_limb_length(self, points: Dict) -> float:
        """Calcola la lunghezza dell'arto in pixel"""
        if points is None or len(points) < 2:
            return 0.0

        try:
            point_names = list(self.landmark_indices[self.limb_type].keys())
            total_length = 0.0

            for i in range(len(point_names) - 1):
                start_name = point_names[i]
                end_name = point_names[i + 1]

                if start_name in points and end_name in points:
                    p1 = np.array(points[start_name])
                    p2 = np.array(points[end_name])
                    segment_length = np.linalg.norm(p2 - p1)
                    total_length += segment_length

            return total_length
        except Exception as e:
            print(f"⚠️ Errore nel calcolo della lunghezza: {e}")
            return 0.0

    def __del__(self):
        """Cleanup resources"""
        if hasattr(self, 'pose') and self.pose is not None and hasattr(self.pose, 'close'):
            try:
                self.pose.close()
            except:
                pass