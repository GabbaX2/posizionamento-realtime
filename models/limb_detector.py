import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
import torch
import landmarker
from landmarker.models import create_model
from landmarker.datasets import AnatomicalLandmarkDataset
import json


class SENIAMLimbDetector:
    def __init__(self, limb_type='leg', model_name='lower_limb'):
        """
        Inizializza il rilevatore con landmarker e regole SENIAM

        Args:
            limb_type: 'arm' o 'leg'
            model_name: nome del modello pre-addestrato di landmarker
        """
        self.limb_type = limb_type
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Carica le regole SENIAM
        self.seniam_rules = self._load_seniam_rules()

        print(f"🔧 Inizializzazione SENIAMLimbDetector per '{limb_type}'...")
        self._initialize_landmarker(model_name)

    def _load_seniam_rules(self) -> Dict:
        """Carica le regole di posizionamento SENIAM"""
        return {
            'vasto_mediale': {
                'landmarks': ['asis', 'patella'],
                'calculation': 'percentage_line',
                'percentage': 50,
                'description': '50% sulla linea tra ASIS e rotula'
            },
            'retto_femorale': {
                'landmarks': ['iliac_spine', 'patella'],
                'calculation': 'percentage_line',
                'percentage': 50,
                'description': '50% sulla linea tra spina iliaca e rotula'
            },
            'gastrocnemio': {
                'landmarks': ['fibular_head', 'malleolus'],
                'calculation': 'percentage_line',
                'percentage': 33,
                'description': '1/3 superiore tra testa fibula e malleolo'
            },
            'tibiale_anteriore': {
                'landmarks': ['tibial_tuberosity', 'malleolus'],
                'calculation': 'percentage_line',
                'percentage': 33,
                'description': '1/3 superiore tra tuberosità tibiale e malleolo'
            }
        }

    def _initialize_landmarker(self, model_name: str):
        """Inizializza il modello landmarker"""
        try:
            print(f"   🧠 Caricamento modello landmarker '{model_name}'...")

            # Opzione 1: Usa un modello pre-addestrato di landmarker
            self.model = create_model(
                model_name=model_name,
                pretrained=True,
                num_classes=len(self._get_expected_landmarks())
            )
            self.model.to(self.device)
            self.model.eval()

            print(f"✅ Landmarker inizializzato su {self.device}")

        except Exception as e:
            print(f"❌ Errore caricamento landmarker: {e}")
            print("🔧 Creando detector semplificato...")
            self._create_fallback_detector()

    def _get_expected_landmarks(self) -> List[str]:
        """Restituisce i landmark attesi per il tipo di arto"""
        if self.limb_type == 'leg':
            return ['asis', 'patella', 'tibial_tuberosity', 'fibular_head', 'malleolus']
        else:  # arm
            return ['acromion', 'lateral_epicondyle', 'medial_epicondyle', 'ulnar_styloid', 'radial_styloid']

    def _create_fallback_detector(self):
        """Crea un detector di fallback quando landmarker non è disponibile"""
        self.model = None
        print("⚠️  Usando detector semplificato - per produzione usa landmarker reale")

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocessa l'immagine per landmarker"""
        if image is None or image.size == 0:
            raise ValueError("Immagine non valida")

        # Converti in RGB
        if len(image.shape) == 3 and image.shape[2] == 3:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            rgb_image = image

        # Normalizza e ridimensiona (landmarker potrebbe richiedere dimensioni specifiche)
        target_size = (512, 512)  # Modifica in base al modello
        processed = cv2.resize(rgb_image, target_size, interpolation=cv2.INTER_AREA)
        processed = processed.astype(np.float32) / 255.0

        # Normalizza (media e std tipiche per ImageNet)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        processed = (processed - mean) / std

        # Converti in tensor PyTorch
        processed = torch.from_numpy(processed).permute(2, 0, 1).unsqueeze(0)

        return processed.to(self.device)

    def detect_landmarks(self, image: np.ndarray) -> Dict[str, Tuple[int, int]]:
        """
        Rileva i landmark anatomici usando landmarker

        Returns:
            Dict con coordinate dei landmark {nome: (x, y)}
        """
        print(f"\n🔍 Rilevamento landmark SENIAM per '{self.limb_type}'...")

        if self.model is None:
            print("⚠️  Usando fallback detection")
            return self._fallback_detection(image)

        try:
            # Preprocessa l'immagine
            input_tensor = self.preprocess_image(image)
            original_h, original_w = image.shape[:2]

            # Inference
            with torch.no_grad():
                outputs = self.model(input_tensor)

            # Estrai coordinate (adatta in base all'output del modello)
            landmarks = self._parse_model_output(outputs, original_w, original_h)

            print(f"✅ {len(landmarks)} landmark rilevati: {list(landmarks.keys())}")
            return landmarks

        except Exception as e:
            print(f"❌ Errore nel detection: {e}")
            return self._fallback_detection(image)

    def _parse_model_output(self, outputs, original_w: int, original_h: int) -> Dict:
        """Analizza l'output del modello per estrarre coordinate"""
        # QUESTA FUNZIONE DIPENDE DAL FORMATO DI OUTPUT DI LANDMARKER
        # Modifica in base al modello specifico che usi

        landmarks = {}

        # Esempio: supponendo che outputs sia un tensor di shape [1, N, 2]
        # dove N è il numero di landmark e 2 sono le coordinate (x, y) normalizzate
        if isinstance(outputs, torch.Tensor) and len(outputs.shape) == 3:
            coords = outputs[0].cpu().numpy()  # [N, 2]

            landmark_names = self._get_expected_landmarks()
            for i, name in enumerate(landmark_names):
                if i < len(coords):
                    x_norm, y_norm = coords[i]
                    x = int(x_norm * original_w)
                    y = int(y_norm * original_h)
                    landmarks[name] = (x, y)

        return landmarks

    def _fallback_detection(self, image: np.ndarray) -> Dict:
        """Detection di fallback quando landmarker non è disponibile"""
        h, w = image.shape[:2]

        # Posizioni approssimative per testing
        if self.limb_type == 'leg':
            return {
                'asis': (int(w * 0.4), int(h * 0.3)),
                'patella': (int(w * 0.45), int(h * 0.6)),
                'tibial_tuberosity': (int(w * 0.46), int(h * 0.65)),
                'fibular_head': (int(w * 0.44), int(h * 0.55)),
                'malleolus': (int(w * 0.48), int(h * 0.85))
            }
        else:  # arm
            return {
                'acromion': (int(w * 0.3), int(h * 0.2)),
                'lateral_epicondyle': (int(w * 0.35), int(h * 0.5)),
                'medial_epicondyle': (int(w * 0.33), int(h * 0.5)),
                'ulnar_styloid': (int(w * 0.4), int(h * 0.8)),
                'radial_styloid': (int(w * 0.42), int(h * 0.8))
            }

    def calculate_sensor_position(self, landmarks: Dict, muscle: str) -> Dict:
        """
        Calcola la posizione del sensore basata sulle regole SENIAM

        Args:
            landmarks: Dict con coordinate landmark
            muscle: Nome del muscolo (es: 'vasto_mediale')

        Returns:
            Dict con posizione e metadati
        """
        if muscle not in self.seniam_rules:
            raise ValueError(f"Muscolo '{muscle}' non supportato. Usa: {list(self.seniam_rules.keys())}")

        rule = self.seniam_rules[muscle]
        required_landmarks = rule['landmarks']

        # Verifica che tutti i landmark richiesti siano disponibili
        missing = [lm for lm in required_landmarks if lm not in landmarks]
        if missing:
            raise ValueError(f"Landmark mancanti per {muscle}: {missing}")

        # Calcola la posizione in base alla regola
        if rule['calculation'] == 'percentage_line':
            start = landmarks[required_landmarks[0]]
            end = landmarks[required_landmarks[1]]
            position = self._calculate_percentage_line(start, end, rule['percentage'])
        else:
            raise ValueError(f"Tipo di calcolo non supportato: {rule['calculation']}")

        return {
            'position': position,
            'muscle': muscle,
            'rule_description': rule['description'],
            'landmarks_used': required_landmarks,
            'confidence': self._calculate_confidence(landmarks, required_landmarks)
        }

    def _calculate_percentage_line(self, start: Tuple[int, int], end: Tuple[int, int],
                                   percentage: float) -> Tuple[int, int]:
        """Calcola un punto a una certa percentuale lungo una linea"""
        x1, y1 = start
        x2, y2 = end

        # Converti percentuale in frazione (es: 50% -> 0.5)
        fraction = percentage / 100.0

        x = int(x1 + fraction * (x2 - x1))
        y = int(y1 + fraction * (y2 - y1))

        return (x, y)

    def _calculate_confidence(self, landmarks: Dict, used_landmarks: List[str]) -> float:
        """Calcola la confidenza della rilevazione"""
        # Implementa una logica di confidenza basata sulla qualità della rilevazione
        # Per ora restituisci un valore fisso
        return 0.85

    def draw_guidance(self, image: np.ndarray, landmarks: Dict,
                      sensor_position: Dict) -> np.ndarray:
        """
        Disegna landmark e posizione del sensore sull'immagine
        """
        result = image.copy()
        h, w = image.shape[:2]

        # 1. Disegna i landmark anatomici
        for name, (x, y) in landmarks.items():
            color = (0, 255, 0)  # Verde
            cv2.circle(result, (x, y), 6, color, 2)
            cv2.putText(result, name, (x + 10, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # 2. Disegna la linea tra i landmark usati
        used_landmarks = sensor_position['landmarks_used']
        if len(used_landmarks) >= 2:
            start = landmarks[used_landmarks[0]]
            end = landmarks[used_landmarks[1]]
            cv2.line(result, start, end, (255, 255, 0), 2, cv2.LINE_AA)

        # 3. Disegna la posizione del sensore
        sensor_x, sensor_y = sensor_position['position']
        cv2.circle(result, (sensor_x, sensor_y), 12, (0, 0, 255), -1)  # Cerchio rosso pieno
        cv2.circle(result, (sensor_x, sensor_y), 15, (255, 255, 255), 3)  # Bordo bianco

        # 4. Aggiungi etichetta
        label = f"Sensore: {sensor_position['muscle']}"
        cv2.putText(result, label, (sensor_x + 20, sensor_y - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 3)
        cv2.putText(result, label, (sensor_x + 20, sensor_y - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # 5. Aggiungi descrizione regola
        desc = sensor_position['rule_description']
        cv2.putText(result, desc, (10, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        return result


