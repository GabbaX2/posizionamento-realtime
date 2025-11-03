import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
import torch
import json

# Import landmarker in modo sicuro con fallback
_LANDMARKER_AVAILABLE = False
try:
    import landmarker

    # Prova diversi pattern di import comuni per librerie di landmark detection
    try:
        from landmarker import create_model

        _LANDMARKER_AVAILABLE = True
    except ImportError:
        try:
            from landmarker.model import create_model

            _LANDMARKER_AVAILABLE = True
        except ImportError:
            try:
                from landmarker import LandmarkDetector as create_model

                _LANDMARKER_AVAILABLE = True
            except ImportError:
                create_model = None
except ImportError:
    create_model = None
    print("⚠️  Package 'landmarker' non trovato - verrà usato il detector di fallback")


class SENIAMLimbDetector:
    def __init__(self, limb_type: str = 'leg', model_name: str = 'lower_limb',
                 target_input_size: Tuple[int, int] = (512, 512)):
        """
        Inizializza il rilevatore con landmarker e regole SENIAM

        Args:
            limb_type: 'arm' o 'leg'
            model_name: nome del modello pre-addestrato di landmarker
            target_input_size: dimensione d'ingresso richiesta dal modello (width, height)
        """
        self.limb_type = limb_type
        self.model = None
        self.model_name = model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.target_input_size = target_input_size  # (W, H) convention used in preprocess

        # Carica le regole SENIAM
        self.seniam_rules = self._load_seniam_rules()

        # Mappa nome modello -> ordine/nome keypoints prodotto dal modello
        self.model_keypoint_map = {
            'lower_limb': ['asis', 'patella', 'tibial_tuberosity', 'fibular_head', 'malleolus'],
            'upper_limb': ['acromion', 'lateral_epicondyle', 'medial_epicondyle', 'ulnar_styloid', 'radial_styloid']
        }

        print(f"🔧 Inizializzazione SENIAMLimbDetector per '{limb_type}' con modello '{model_name}'...")
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
                'landmarks': ['asis', 'patella'],
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
        if not _LANDMARKER_AVAILABLE or create_model is None:
            print("⚠️  Landmarker non disponibile: useremo il detector di fallback")
            self._create_fallback_detector()
            return

        try:
            print(f"   🧠 Caricamento modello landmarker '{model_name}'...")

            # Prova diversi pattern di inizializzazione comuni
            try:
                # Pattern 1: factory function con parametri
                self.model = create_model(
                    model_name=model_name,
                    pretrained=True,
                    num_classes=len(self._get_expected_landmarks())
                )
            except TypeError:
                try:
                    # Pattern 2: solo model_name
                    self.model = create_model(model_name=model_name)
                except TypeError:
                    # Pattern 3: istanza diretta
                    self.model = create_model()

            if hasattr(self.model, 'to'):
                self.model.to(self.device)
            if hasattr(self.model, 'eval'):
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

    def preprocess_image(self, image: np.ndarray, target_size: Tuple[int, int] = None) -> Tuple[torch.Tensor, dict]:
        """
        Preprocessa l'immagine per landmarker restituendo anche meta per undo letterbox.
        Usa letterbox (pad) per mantenere aspect ratio.
        target_size: (width, height)
        """
        if image is None or image.size == 0:
            raise ValueError("Immagine non valida")

        if target_size is None:
            target_size = self.target_input_size
        tw, th = target_size
        h, w = image.shape[:2]

        # Calcola scale e dimensioni ridimensionate
        scale = min(tw / w, th / h)
        nw, nh = int(w * scale), int(h * scale)

        resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)

        # Crea canvas e incolla centrato (letterbox)
        canvas = np.zeros((th, tw, 3), dtype=np.uint8)
        dx = (tw - nw) // 2
        dy = (th - nh) // 2
        canvas[dy:dy + nh, dx:dx + nw] = resized

        # Converti in RGB e normalizza
        if canvas.ndim == 3 and canvas.shape[2] == 3:
            rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        else:
            rgb = canvas.astype(np.float32) / 255.0

        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        inp = (rgb - mean) / std

        tensor = torch.from_numpy(inp).permute(2, 0, 1).unsqueeze(0).to(self.device)

        meta = {'orig_shape': (h, w), 'input_shape': (th, tw), 'scale': scale, 'dx': dx, 'dy': dy}
        return tensor, meta

    def detect_landmarks(self, image: np.ndarray, score_threshold: float = 0.2) -> Dict[str, Dict]:
        """
        Rileva i landmark anatomici usando landmarker

        Returns:
            Dict con struttura:
            { 'asis': {'xy': (x,y) or None, 'score': float}, ... }
        """
        print(f"\n🔍 Rilevamento landmark SENIAM per '{self.limb_type}'...")

        if self.model is None:
            print("⚠️  Usando fallback detection")
            # Convert fallback simple coords to standard format
            fallback = self._fallback_detection(image)
            formatted = {}
            for k, v in fallback.items():
                formatted[k] = {'xy': v, 'score': 0.5}
            return formatted

        try:
            input_tensor, meta = self.preprocess_image(image)

            with torch.no_grad():
                outputs = self.model(input_tensor)

            landmarks = self._parse_model_output(outputs, meta, score_threshold)

            detected_names = [k for k, v in landmarks.items() if v['xy'] is not None]
            print(f"✅ {len(detected_names)} landmark rilevati con score>={score_threshold}: {detected_names}")
            return landmarks

        except Exception as e:
            print(f"❌ Errore nel detection: {e}")
            # fallback strutturato
            fallback = self._fallback_detection(image)
            formatted = {}
            for k, v in fallback.items():
                formatted[k] = {'xy': v, 'score': 0.5}
            return formatted

    def _parse_model_output(self, outputs, meta: dict, score_threshold: float = 0.2) -> Dict[str, Dict]:
        """
        Analizza l'output del modello e converte in coordinate immagine con score.
        Supporta:
        - heatmaps tensor [1, N, H, W]
        - coords tensor [1, N, 2] (valori normalizzati 0..1)
        - dict con chiavi 'heatmaps' o 'coords'
        """
        # Ottieni nomi expected dal mapping modello -> keypoints
        landmark_names = self.model_keypoint_map.get(self.model_name, self._get_expected_landmarks())

        coords_norm = None  # Nx2 array valori in input image space (0..1)
        scores = None

        # Caso dict con heatmaps o coords
        if isinstance(outputs, dict):
            # Heatmaps preferite
            if 'heatmaps' in outputs and isinstance(outputs['heatmaps'], torch.Tensor):
                coords_norm, scores = self._postprocess_heatmaps(outputs['heatmaps'])
            # Altra possibile chiave
            elif 'preds' in outputs and isinstance(outputs['preds'], torch.Tensor) and outputs['preds'].dim() == 4:
                coords_norm, scores = self._postprocess_heatmaps(outputs['preds'])
            elif 'coords' in outputs and isinstance(outputs['coords'], torch.Tensor):
                coords_norm = outputs['coords'][0].cpu().numpy()
                # se esiste 'scores' in outputs usalo
                if 'scores' in outputs and isinstance(outputs['scores'], torch.Tensor):
                    scores = outputs['scores'][0].cpu().numpy()
                else:
                    scores = np.ones(len(coords_norm), dtype=np.float32) * 0.8
        # Caso tensor
        elif isinstance(outputs, torch.Tensor):
            # heatmaps [1,N,H,W]
            if outputs.dim() == 4:
                coords_norm, scores = self._postprocess_heatmaps(outputs)
            # coords [1,N,2]
            elif outputs.dim() == 3 and outputs.shape[2] == 2:
                coords_norm = outputs[0].cpu().numpy()
                scores = np.ones(len(coords_norm), dtype=np.float32) * 0.8
            else:
                # formato non riconosciuto
                coords_norm = None

        # Se non siamo riusciti a ricavare coords, ritorna struttura vuota con None
        if coords_norm is None:
            result = {}
            for name in landmark_names:
                result[name] = {'xy': None, 'score': 0.0}
            return result

        # Converti coords normalizzate rispetto all'input (letterboxed) nello spazio immagine originale
        coords_img = self._coords_to_image_space(coords_norm, meta)

        result: Dict[str, Dict] = {}
        for i, name in enumerate(landmark_names):
            if i < len(coords_img):
                xy = coords_img[i]
                score = float(scores[i]) if (scores is not None and i < len(scores)) else 0.0
                if score < score_threshold:
                    result[name] = {'xy': None, 'score': float(score)}
                else:
                    result[name] = {'xy': xy, 'score': float(score)}
            else:
                result[name] = {'xy': None, 'score': 0.0}

        return result

    def _postprocess_heatmaps(self, heatmaps: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estrae per ogni heatmap la posizione del massimo e la normalizza.
        heatmaps: tensor [1, N, H, W] o [N, H, W] -> ritorna coords_norm Nx2 e scores N
        coords_norm in input-image-space (0..1) relative alla dimensione della input_shape (letterboxed)
        """
        hm = heatmaps.cpu()
        if hm.dim() == 4 and hm.shape[0] == 1:
            hm = hm[0]  # [N, H, W]
        elif hm.dim() == 3:
            pass
        else:
            raise ValueError("Formato heatmaps non supportato")

        N, H, W = hm.shape
        coords = np.zeros((N, 2), dtype=np.float32)
        scores = np.zeros((N,), dtype=np.float32)

        hm_np = hm.numpy()
        for i in range(N):
            flat_idx = np.argmax(hm_np[i])
            y, x = divmod(flat_idx, W)
            score = float(hm_np[i, y, x])
            # centro del pixel convertito in coordinata normalizzata rispetto alla input image
            coords[i, 0] = (x + 0.5) / W
            coords[i, 1] = (y + 0.5) / H
            scores[i] = score

        return coords, scores

    def _coords_to_image_space(self, coords_norm: np.ndarray, meta: dict) -> List[Tuple[int, int]]:
        """
        Converte coords normalizzate nell'input (letterboxed) alla scala dell'immagine originale.
        coords_norm: Nx2 con valori tra 0 e 1 relativi alla input_shape (th, tw)
        meta: contiene orig_shape, input_shape, scale, dx, dy
        """
        h, w = meta['orig_shape']
        th, tw = meta['input_shape']
        dx = meta['dx']
        dy = meta['dy']
        scale = meta['scale']

        out = []
        for (xn, yn) in coords_norm:
            # posizione nello spazio input (letterbox)
            x_in = xn * tw
            y_in = yn * th
            # togli offset letterbox e riporta su immagine originale
            x_unpad = (x_in - dx) / scale
            y_unpad = (y_in - dy) / scale
            x_unpad = int(round(x_unpad))
            y_unpad = int(round(y_unpad))
            # clamp per sicurezza
            x_unpad = max(0, min(w - 1, x_unpad))
            y_unpad = max(0, min(h - 1, y_unpad))
            out.append((x_unpad, y_unpad))
        return out

    def _fallback_detection(self, image: np.ndarray) -> Dict:
        """Detection di fallback quando landmarker non è disponibile (coordinate raw)"""
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

    def calculate_sensor_position(self, landmarks: Dict[str, Dict], muscle: str,
                                  min_score: float = 0.2) -> Dict:
        """
        Calcola la posizione del sensore basata sulle regole SENIAM

        Args:
            landmarks: Dict con struttura {name: {'xy': (x,y) or None, 'score': float}}
            muscle: Nome del muscolo (es: 'vasto_mediale')
            min_score: soglia minima di confidenza per i landmark

        Returns:
            Dict con posizione e metadati
        """
        if muscle not in self.seniam_rules:
            raise ValueError(f"Muscolo '{muscle}' non supportato. Usa: {list(self.seniam_rules.keys())}")

        rule = self.seniam_rules[muscle]
        required_landmarks = rule['landmarks']

        # Verifica che tutti i landmark richiesti siano presenti e sopra soglia
        missing = [lm for lm in required_landmarks
                   if lm not in landmarks or landmarks[lm].get('xy') is None
                   or landmarks[lm].get('score', 0.0) < min_score]
        if missing:
            raise ValueError(f"Landmark mancanti o a bassa confidenza per {muscle}: {missing}")

        # Estrai coordinate reali
        start = landmarks[required_landmarks[0]]['xy']
        end = landmarks[required_landmarks[1]]['xy']

        # Calcola posizione in base alla regola
        if rule['calculation'] == 'percentage_line':
            position = self._calculate_percentage_line(start, end, rule['percentage'])
        else:
            raise ValueError(f"Tipo di calcolo non supportato: {rule['calculation']}")

        # Confidenza aggregata = media dei landmark usati
        scores = [landmarks[lm]['score'] for lm in required_landmarks]
        confidence = float(sum(scores) / len(scores)) if scores else 0.0

        return {
            'position': position,
            'muscle': muscle,
            'rule_description': rule['description'],
            'landmarks_used': required_landmarks,
            'confidence': confidence
        }

    def _calculate_percentage_line(self, start: Tuple[int, int], end: Tuple[int, int],
                                   percentage: float) -> Tuple[int, int]:
        """Calcola un punto a una certa percentuale lungo una linea"""
        x1, y1 = start
        x2, y2 = end
        fraction = percentage / 100.0
        x = int(round(x1 + fraction * (x2 - x1)))
        y = int(round(y1 + fraction * (y2 - y1)))
        return (x, y)

    def draw_guidance(self, image: np.ndarray, landmarks: Dict[str, Dict],
                      sensor_position: Dict, low_conf_threshold: float = 0.2) -> np.ndarray:
        """
        Disegna landmark e posizione del sensore sull'immagine.

        Args:
            image: immagine originale
            landmarks: {name: {'xy': (x,y)|None, 'score': float}}
            sensor_position: dict restituito da calculate_sensor_position
            low_conf_threshold: soglia per colorare i punti a bassa confidenza
        """
        result = image.copy()
        h, w = image.shape[:2]

        # 1. Disegna i landmark anatomici
        for name, info in landmarks.items():
            xy = info.get('xy')
            score = info.get('score', 0.0)
            if xy is None:
                continue
            x, y = xy
            # Colore in base a confidenza: verde (alta), giallo (media), rosso (bassa)
            if score >= 0.75:
                color = (0, 255, 0)
            elif score >= 0.4:
                color = (0, 255, 255)
            else:
                color = (0, 0, 255)
            cv2.circle(result, (x, y), 6, color, 2)
            cv2.putText(result, f"{name}:{score:.2f}", (x + 8, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)

        # 2. Disegna la linea tra i landmark usati (se presenti)
        used_landmarks = sensor_position.get('landmarks_used', [])
        if len(used_landmarks) >= 2:
            lm0 = landmarks.get(used_landmarks[0], {})
            lm1 = landmarks.get(used_landmarks[1], {})
            if lm0.get('xy') is not None and lm1.get('xy') is not None:
                start = lm0['xy']
                end = lm1['xy']
                cv2.line(result, start, end, (255, 255, 0), 2, cv2.LINE_AA)

        # 3. Disegna la posizione del sensore
        sensor_xy = sensor_position.get('position')
        if sensor_xy is not None:
            sensor_x, sensor_y = sensor_xy
            cv2.circle(result, (sensor_x, sensor_y), 12, (0, 0, 255), -1)  # Cerchio rosso pieno
            cv2.circle(result, (sensor_x, sensor_y), 15, (255, 255, 255), 3)  # Bordo bianco

            # 4. Aggiungi etichetta
            label = f"Sensore: {sensor_position.get('muscle', '?')} ({sensor_position.get('confidence', 0.0):.2f})"
            cv2.putText(result, label, (sensor_x + 20, sensor_y - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 3)
            cv2.putText(result, label, (sensor_x + 20, sensor_y - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # 5. Aggiungi descrizione regola
        desc = sensor_position.get('rule_description', '')
        cv2.putText(result, desc, (10, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        return result