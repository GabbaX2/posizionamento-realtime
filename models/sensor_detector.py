import cv2
import numpy as np
from typing import List, Dict, Tuple
import json
import os


class SensorDetector:
    def __init__(self, config_path: str = 'data/calibration/color_calibration.json',
                 detection_mode: str = 'electrodes'):
        """
        Initialize sensor detector

        Args:
            config_path: Path to color calibration file
            detection_mode: 'electrodes' for medical electrodes, 'colored' for colored sensors
        """
        self.config_path = config_path
        self.detection_mode = detection_mode
        self.color_ranges = self._load_color_ranges()
        print(f"🎨 SensorDetector inizializzato in modalità: {detection_mode}")

    def _load_color_ranges(self) -> Dict:
        """Load color ranges from configuration file"""
        default_ranges = {
            'white': [
                {'lower': [0, 0, 200], 'upper': [180, 50, 255]}
            ],
            'silver': [
                {'lower': [0, 0, 180], 'upper': [180, 30, 255]}
            ],
            'light_blue': [
                {'lower': [85, 30, 150], 'upper': [105, 255, 255]},
                {'lower': [80, 20, 180], 'upper': [110, 120, 255]}
            ],
            'cyan': [
                {'lower': [75, 50, 150], 'upper': [95, 255, 255]}
            ]
        }

        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    loaded_ranges = json.load(f)
                    print(f"✅ Range colori caricati da: {self.config_path}")
                    for key, value in loaded_ranges.items():
                        default_ranges[key] = value
        except Exception as e:
            print(f"⚠️ Errore nel caricamento range colori: {e}")

        return default_ranges

    def detect_sensors(self, image: np.ndarray, min_area: int = 200, max_area: int = 8000,
                       min_circularity: float = 0.5, max_sensors: int = 10) -> List[Dict]:
        """
        Detect sensors/electrodes in image
        """
        print(f"\n🔍 Rilevamento {'elettrodi' if self.detection_mode == 'electrodes' else 'sensori'}...")

        if self.detection_mode == 'electrodes':
            return self._detect_medical_electrodes(image, min_area, max_area, min_circularity, max_sensors)
        else:
            return self._detect_colored_sensors(image, min_area, max_area, min_circularity, max_sensors)

    def detect_sensors_realtime(self, image: np.ndarray) -> List[Dict]:
        """
        Rilevamento elettrodi in tempo reale con massima precisione.
        Disegna un cerchio attorno all'elettrodo e lo segue mentre si muove.
        """
        h, w = image.shape[:2]

        # Range più ampio per adattarsi a elettrodi di dimensioni sconosciute
        min_r = max(6, int(min(h, w) * 0.005))
        max_r = int(min(h, w) * 0.18)  # fino al 18% del lato minore

        electrodes = []

        # 1. Metodo centro metallico (più affidabile per Kendall)
        metallic = self._detect_metallic_center_electrodes(image, min_r, max_r)

        # 2. Metodo bright circles (fallback)
        bright = self._detect_bright_circles_realtime(image, min_r, max_r)

        # 3. Hough come ultima risorsa
        hough = []
        if not metallic and not bright:
            hough = self._detect_electrodes_hough(image, min_r, max_r, fast_mode=True)

        # Combina risultati
        all_detections = metallic + bright + hough

        # Rimuovi duplicati
        electrodes = self._remove_duplicate_sensors(all_detections, distance_threshold=25)

        # Filtra con soglia bassa per non perdere l’elettrodo
        electrodes = [e for e in electrodes if e['confidence'] > 0.25]

        # Ordina per confidenza
        electrodes = sorted(electrodes, key=lambda s: s['confidence'], reverse=True)

        # Mantieni solo il più probabile (quello che tieni in mano)
        if electrodes:
            electrodes = [electrodes[0]]

        return electrodes

    def _detect_medical_electrodes(self, image: np.ndarray, min_area: int, max_area: int,
                                   min_circularity: float, max_sensors: int) -> List[Dict]:
        """
        Detect medical electrodes with improved accuracy - VERSIONE MIGLIORATA
        """
        print("   🏥 Modalità rilevamento elettrodi medicali (MIGLIORATA)")

        h, w = image.shape[:2]
        min_r = max(10, int(min(h, w) * 0.008))
        max_r = int(min(h, w) * 0.045)

        print(f"📐 Range Raggio (Pixel): [{min_r}, {max_r}]")

        electrodes = []

        # NUOVO: Metodo per elettrodi con centro metallico/argentato
        electrodes_metallic = self._detect_metallic_center_electrodes(image, min_r, max_r)

        # Metodi esistenti con soglie più rilassate
        electrodes_hough = self._detect_electrodes_hough(image, min_r, max_r)
        electrodes_bright = self._detect_bright_circles(image, min_r, max_r)
        electrodes_structured = self._detect_structured_electrodes(image, min_r, max_r)

        # Combine results
        all_detections = electrodes_metallic + electrodes_bright + electrodes_hough + electrodes_structured

        print(f"   📊 Rilevamenti grezzi: metallic={len(electrodes_metallic)}, "
              f"bright={len(electrodes_bright)}, hough={len(electrodes_hough)}, "
              f"structured={len(electrodes_structured)}")

        # Remove duplicates
        electrodes = self._remove_duplicate_sensors(all_detections, distance_threshold=30)

        # Filtro meno restrittivo
        electrodes = [e for e in electrodes if e['confidence'] > 0.45]  # Ridotto da 0.65

        # Validate electrodes
        electrodes = self._validate_electrodes(image, electrodes)

        # Limit maximum number
        if len(electrodes) > max_sensors:
            electrodes = sorted(electrodes, key=lambda s: s['confidence'], reverse=True)[:max_sensors]

        print(f"✅ Elettrodi rilevati: {len(electrodes)}")
        for i, electrode in enumerate(electrodes):
            print(f"   • Elettrodo {i + 1}: pos={electrode['position']}, "
                  f"confidence={electrode['confidence']:.2f}, Raggio={electrode.get('radius', 'N/A')}")

        return electrodes

    def _detect_metallic_center_electrodes(self, image: np.ndarray, min_radius_pixel: int,
                                           max_radius_pixel: int) -> List[Dict]:
        """
        NUOVO METODO: Rileva elettrodi con centro metallico/argentato
        Specifico per elettrodi come quello nell'immagine
        """
        detected = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # 1. Trova regioni metalliche/argentate (grigio chiaro con bassa saturazione)
        # Centro argentato: alta luminosità (150-230), bassa saturazione
        lower_metallic = np.array([0, 0, 150])  # Qualsiasi tonalità, bassa saturazione, luminosità media-alta
        upper_metallic = np.array([180, 40, 230])  # Soglia più bassa per catturare metallico

        metallic_mask = cv2.inRange(hsv, lower_metallic, upper_metallic)

        # Applica morphology per pulire
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        metallic_mask = cv2.morphologyEx(metallic_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        metallic_mask = cv2.morphologyEx(metallic_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        # Trova contorni dei centri metallici
        contours, _ = cv2.findContours(metallic_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        min_center_area = np.pi * (min_radius_pixel * 0.15) ** 2  # Centro più piccolo
        max_center_area = np.pi * (max_radius_pixel * 0.5) ** 2

        for contour in contours:
            area = cv2.contourArea(contour)

            if area < min_center_area or area > max_center_area:
                continue

            # Verifica circolarità del centro metallico
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                if circularity < 0.5:  # Meno restrittivo
                    continue
            else:
                continue

            # Trova centro
            M = cv2.moments(contour)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                continue

            # Stima raggio elettrodo completo (centro metallico è ~20% del totale)
            (_, _), center_radius = cv2.minEnclosingCircle(contour)
            estimated_radius = int(center_radius * 4.0)  # Il centro è circa 1/4 del raggio totale

            if not (min_radius_pixel < estimated_radius < max_radius_pixel):
                continue

            # 2. Verifica presenza anello blu intorno
            r_int = estimated_radius
            y1 = max(0, cy - r_int)
            y2 = min(image.shape[0], cy + r_int)
            x1 = max(0, cx - r_int)
            x2 = min(image.shape[1], cx + r_int)

            roi_hsv = hsv[y1:y2, x1:x2]

            if roi_hsv.size == 0:
                continue

            # Cerca anello blu/azzurro
            has_blue = self._check_blue_ring_around_center(roi_hsv, (cx - x1, cy - y1), center_radius, r_int)

            # 3. Verifica contrasto con sfondo
            roi_gray = gray[y1:y2, x1:x2]
            if roi_gray.size > 0:
                # Il centro deve essere più chiaro della media dello sfondo
                center_brightness = gray[cy, cx]
                bg_brightness = np.mean(roi_gray)
                contrast = center_brightness - bg_brightness

                if contrast < 10:  # Deve esserci contrasto
                    continue

            # Calcola confidence
            confidence = circularity * 0.4
            if has_blue:
                confidence += 0.3
            if contrast > 30:
                confidence += 0.2
            else:
                confidence += 0.1

            detected.append({
                'position': (cx, cy),
                'bbox': (x1, y1, x2 - x1, y2 - y1),
                'color': 'electrode',
                'area': (x2 - x1) * (y2 - y1),
                'circularity': circularity,
                'aspect_ratio': 1.0,
                'confidence': min(confidence, 1.0),
                'method': 'metallic',
                'radius': r_int
            })

        print(f"   🔧 Metodo metallico: {len(detected)} candidati")
        return detected

    def _check_blue_ring_around_center(self, roi_hsv: np.ndarray, center_local: Tuple[int, int],
                                       inner_radius: float, outer_radius: int) -> bool:
        """
        Verifica presenza anello blu/azzurro intorno al centro metallico
        """
        if roi_hsv.size == 0:
            return False

        h, w = roi_hsv.shape[:2]
        cx_local, cy_local = center_local

        # Assicurati che il centro sia dentro la ROI
        if cx_local < 0 or cy_local < 0 or cx_local >= w or cy_local >= h:
            return False

        # Crea maschera anello (tra centro metallico e bordo esterno)
        ring_mask = np.zeros((h, w), dtype=np.uint8)

        inner_r = max(1, int(inner_radius * 1.2))  # Poco fuori dal centro metallico
        outer_r = min(min(h, w) // 2, int(outer_radius * 0.7))  # Parte media dell'elettrodo

        if outer_r <= inner_r:
            return False

        cv2.circle(ring_mask, (cx_local, cy_local), outer_r, 255, -1)
        cv2.circle(ring_mask, (cx_local, cy_local), inner_r, 0, -1)

        ring_pixels = roi_hsv[ring_mask > 0]

        if len(ring_pixels) == 0:
            return False

        # Cerca pixel blu/azzurri nell'anello
        # Blu/Azzurro: H tra 85-110, S > 20, V > 50
        blue_mask = ((ring_pixels[:, 0] >= 85) & (ring_pixels[:, 0] <= 110) &
                     (ring_pixels[:, 1] > 20) & (ring_pixels[:, 2] > 50))

        blue_ratio = np.sum(blue_mask) / len(ring_pixels)

        return blue_ratio > 0.10  # Almeno 10% di pixel blu nell'anello

    def _detect_bright_circles(self, image: np.ndarray, min_radius_pixel: int,
                               max_radius_pixel: int, fast_mode: bool = False) -> List[Dict]:
        """Detect bright circular regions - SOGLIE RILASSATE"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # SOGLIA PIÙ BASSA per catturare centri metallici
        threshold = 170 if not fast_mode else 160  # Era 210/200
        _, bright_mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

        kernel_size = 3 if fast_mode else 5
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        iterations = 2 if fast_mode else 3

        bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_OPEN, kernel, iterations=iterations)
        bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(bright_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        min_center_area = np.pi * (min_radius_pixel * 0.2) ** 2
        max_center_area = np.pi * (max_radius_pixel * 0.7) ** 2

        detected = []
        for contour in contours:
            area = cv2.contourArea(contour)

            if area < min_center_area or area > max_center_area:
                continue

            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                min_circ = 0.55 if fast_mode else 0.60  # Meno restrittivo
                if circularity < min_circ:
                    continue
            else:
                continue

            M = cv2.moments(contour)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                (_, _), radius = cv2.minEnclosingCircle(contour)
                estimated_radius = int(radius * 3.5)

                if not (min_radius_pixel < estimated_radius < max_radius_pixel):
                    continue

                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = float(w) / h if h > 0 else 0

                min_aspect = 0.65 if fast_mode else 0.70  # Meno restrittivo
                max_aspect = 1.35 if fast_mode else 1.30

                if not (min_aspect < aspect_ratio < max_aspect):
                    continue

                confidence = circularity * 0.6

                detected.append({
                    'position': (cx, cy),
                    'bbox': (x, y, w, h),
                    'color': 'electrode',
                    'area': area,
                    'circularity': circularity,
                    'aspect_ratio': aspect_ratio,
                    'confidence': confidence,
                    'method': 'bright',
                    'radius': estimated_radius
                })

        return detected

    def _detect_bright_circles_realtime(self, image: np.ndarray, min_radius_pixel: int,
                                        max_radius_pixel: int) -> List[Dict]:
        """
        Versione ULTRA-VELOCE per tempo reale
        Parametri molto rilassati per catturare elettrodi in movimento
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Soglia MOLTO BASSA per centri metallici/argentati
        threshold = 140  # Era 200+
        _, bright_mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

        # Morphology minima per velocità
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        contours, _ = cv2.findContours(bright_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Area molto permissiva
        min_center_area = np.pi * (min_radius_pixel * 0.15) ** 2
        max_center_area = np.pi * (max_radius_pixel * 0.8) ** 2

        detected = []
        for contour in contours:
            area = cv2.contourArea(contour)

            if area < min_center_area or area > max_center_area:
                continue

            # Circolarità MOLTO rilassata
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                if circularity < 0.4:  # Era 0.6+
                    continue
            else:
                continue

            M = cv2.moments(contour)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                (_, _), radius = cv2.minEnclosingCircle(contour)
                # Stima conservativa del raggio totale
                estimated_radius = int(radius * 3.0)

                if not (min_radius_pixel < estimated_radius < max_radius_pixel):
                    continue

                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = float(w) / h if h > 0 else 0

                # Aspect ratio MOLTO rilassato
                if not (0.5 < aspect_ratio < 2.0):  # Era 0.7-1.3
                    continue

                # NO texture check per velocità

                confidence = circularity * 0.8 + 0.2  # Boost iniziale

                detected.append({
                    'position': (cx, cy),
                    'bbox': (x, y, w, h),
                    'color': 'electrode',
                    'area': area,
                    'circularity': circularity,
                    'aspect_ratio': aspect_ratio,
                    'confidence': confidence,
                    'method': 'bright_rt',
                    'radius': estimated_radius
                })

        return detected

    def _remove_background_artifacts(self, image: np.ndarray) -> np.ndarray:
        """
        Rimuove artefatti tipici della rimozione sfondo (pixel bianchi sparsi, bordi irregolari)
        """
        # Converti in grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Crea maschera delle zone "molto chiare" (potenziali artefatti)
        _, bright_mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)

        # Rimuovi piccoli cluster bianchi (artefatti)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_OPEN, kernel, iterations=2)

        # Trova contorni dei blob bianchi
        contours, _ = cv2.findContours(bright_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Crea maschera degli artefatti da rimuovere
        artifact_mask = np.zeros_like(gray)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Se il blob è troppo piccolo o ha forma irregolare → è un artefatto
            if area < 50:  # Blob troppo piccolo
                cv2.drawContours(artifact_mask, [cnt], -1, 255, -1)
            else:
                perimeter = cv2.arcLength(cnt, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter ** 2)
                    # Se non è circolare → probabilmente artefatto
                    if circularity < 0.4:
                        cv2.drawContours(artifact_mask, [cnt], -1, 255, -1)

        # Sostituisci gli artefatti con grigio medio
        cleaned = image.copy()
        cleaned[artifact_mask > 0] = [128, 128, 128]

        return cleaned

    def _filter_background_artifacts(self, image: np.ndarray, electrodes: List[Dict]) -> List[Dict]:
        """
        Filtra elettrodi che sono probabilmente artefatti da sfondo
        Verifica il contesto intorno all'elettrodo
        """
        filtered = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        for electrode in electrodes:
            cx, cy = electrode['position']
            radius = electrode.get('radius', 20)

            # Espandi la ROI per verificare il contesto
            context_radius = int(radius * 2.5)
            y1 = max(0, cy - context_radius)
            y2 = min(h, cy + context_radius)
            x1 = max(0, cx - context_radius)
            x2 = min(w, cx + context_radius)

            context_roi = gray[y1:y2, x1:x2]

            if context_roi.size == 0:
                continue

            # 1. Verifica se c'è abbastanza contrasto con lo sfondo
            mean_intensity = np.mean(context_roi)
            center_intensity = gray[cy, cx]

            contrast = abs(center_intensity - mean_intensity)

            # Se il contrasto è basso, probabilmente è un artefatto sullo sfondo uniforme
            if contrast < 30:
                print(f"   ⚠️ Scartato elettrodo a {(cx, cy)} - basso contrasto: {contrast:.1f}")
                continue

            # 2. Verifica la presenza di texture cutanea intorno
            # La pelle ha texture, lo sfondo trasparente no
            laplacian = cv2.Laplacian(context_roi, cv2.CV_64F)
            texture_variance = np.var(laplacian)

            # Se la varianza di texture è troppo bassa → sfondo uniforme
            if texture_variance < 50:
                print(f"   ⚠️ Scartato elettrodo a {(cx, cy)} - texture insufficiente: {texture_variance:.1f}")
                continue

            # 3. Verifica che non sia in una zona completamente bianca/grigia (bordi immagine)
            if mean_intensity > 230 or mean_intensity < 30:
                print(f"   ⚠️ Scartato elettrodo a {(cx, cy)} - zona non valida (intensità: {mean_intensity:.1f})")
                continue

            # 4. Verifica presenza di bordi definiti
            edges = cv2.Canny(context_roi, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size

            # Se ci sono troppi pochi bordi → probabilmente artefatto su sfondo piatto
            if edge_density < 0.02:
                print(f"   ⚠️ Scartato elettrodo a {(cx, cy)} - bordi insufficienti: {edge_density:.4f}")
                continue

            # Elettrodo valido
            filtered.append(electrode)

        print(f"   🔍 Filtro artefatti: {len(electrodes)} → {len(filtered)} elettrodi")
        return filtered

    def _filter_by_size_consistency(self, electrodes: List[Dict]) -> List[Dict]:
        """
        Filtra elettrodi con dimensioni anomale rispetto alla media
        """
        if len(electrodes) < 2:
            return electrodes

        radii = [e.get('radius', 0) for e in electrodes if 'radius' in e]

        if not radii:
            return electrodes

        median_radius = np.median(radii)

        # Accetta solo elettrodi con raggio entro ±50% della mediana
        filtered = []
        for electrode in electrodes:
            radius = electrode.get('radius', median_radius)

            ratio = radius / median_radius if median_radius > 0 else 1.0

            if 0.5 <= ratio <= 1.5:
                filtered.append(electrode)
            else:
                cx, cy = electrode['position']
                print(
                    f"   ⚠️ Scartato elettrodo a {(cx, cy)} - raggio anomalo: {radius} (mediana: {median_radius:.1f})")

        return filtered

    def _detect_bright_circles(self, image: np.ndarray, min_radius_pixel: int,
                               max_radius_pixel: int, fast_mode: bool = False) -> List[Dict]:
        """Detect bright circular regions (electrode centers) - IMPROVED"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # SOGLIA PIÙ ALTA per evitare artefatti
        threshold = 210 if not fast_mode else 200  # Era 200/190
        _, bright_mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

        kernel_size = 3 if fast_mode else 5
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        iterations = 2 if fast_mode else 3  # Più iterazioni per pulizia

        bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_OPEN, kernel, iterations=iterations)
        bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(bright_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        min_center_area = np.pi * (min_radius_pixel * 0.25) ** 2  # Era 0.2
        max_center_area = np.pi * (max_radius_pixel * 0.6) ** 2  # Era 0.7

        detected = []
        for contour in contours:
            area = cv2.contourArea(contour)

            if area < min_center_area or area > max_center_area:
                continue

            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                min_circ = 0.65 if fast_mode else 0.75  # PIÙ RESTRITTIVO (era 0.55/0.65)
                if circularity < min_circ:
                    continue
            else:
                continue

            M = cv2.moments(contour)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                (_, _), radius = cv2.minEnclosingCircle(contour)
                estimated_radius = int(radius * 3.5)

                if not (min_radius_pixel < estimated_radius < max_radius_pixel):
                    continue

                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = float(w) / h if h > 0 else 0

                min_aspect = 0.75 if fast_mode else 0.80  # PIÙ RESTRITTIVO (era 0.70/0.75)
                max_aspect = 1.25 if fast_mode else 1.20  # PIÙ RESTRITTIVO (era 1.30/1.25)

                if not (min_aspect < aspect_ratio < max_aspect):
                    continue

                # Texture check SEMPRE attivo (anche in fast mode)
                roi = gray[y:y + h, x:x + w]
                if roi.size > 0:
                    std_dev = np.std(roi)
                    if std_dev > 25:  # Era 30
                        continue

                confidence = circularity * 0.7

                detected.append({
                    'position': (cx, cy),
                    'bbox': (x, y, w, h),
                    'color': 'electrode',
                    'area': area,
                    'circularity': circularity,
                    'aspect_ratio': aspect_ratio,
                    'confidence': confidence,
                    'method': 'bright',
                    'radius': estimated_radius
                })

        return detected

    def _detect_electrodes_hough(self, image: np.ndarray, min_radius_pixel: int,
                                 max_radius_pixel: int, fast_mode: bool = False) -> List[Dict]:
        """Hough circle detection - IMPROVED"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        blur_kernel = (7, 7) if fast_mode else (9, 9)
        blurred = cv2.GaussianBlur(gray, blur_kernel, 2)

        edges = cv2.Canny(blurred, 50, 150)

        param2 = 40 if fast_mode else 45  # PIÙ RESTRITTIVO (era 30/35)
        min_dist = 40 if fast_mode else 50  # DISTANZA MAGGIORE (era 35/40)

        circles = cv2.HoughCircles(
            edges,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=min_dist,
            param1=50,
            param2=param2,
            minRadius=min_radius_pixel,
            maxRadius=max_radius_pixel
        )

        detected = []
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for circle in circles[0, :]:
                cx, cy, r = circle

                # Brightness check SEMPRE attivo
                if gray[cy, cx] < 210:  # Era 200
                    continue

                y1, y2 = max(0, cy - r // 2), min(gray.shape[0], cy + r // 2)
                x1, x2 = max(0, cx - r // 2), min(gray.shape[1], cx + r // 2)
                roi = gray[y1:y2, x1:x2]

                if roi.size > 0:
                    std_dev = np.std(roi)
                    if std_dev > 20:  # Era 25
                        continue

                area = np.pi * r * r
                detected.append({
                    'position': (int(cx), int(cy)),
                    'bbox': (int(cx - r), int(cy - r), int(2 * r), int(2 * r)),
                    'color': 'electrode',
                    'area': area,
                    'circularity': 1.0,
                    'aspect_ratio': 1.0,
                    'confidence': 0.6,
                    'method': 'hough',
                    'radius': int(r)
                })

        return detected

    def _detect_structured_electrodes(self, image: np.ndarray, min_radius_pixel: int,
                                      max_radius_pixel: int) -> List[Dict]:
        """
        Detect electrodes based on structure: white center + blue/dark border
        """
        detected = []
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        _, bright_mask = cv2.threshold(gray, 215, 255, cv2.THRESH_BINARY)  # Era 210

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_OPEN, kernel)
        bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(bright_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        min_center_area = np.pi * (min_radius_pixel * 0.25) ** 2  # Era 0.2
        max_center_area = np.pi * (max_radius_pixel * 0.6) ** 2  # Era 0.7

        for contour in contours:
            area = cv2.contourArea(contour)

            if area < min_center_area or area > max_center_area:
                continue

            (cx, cy), radius_approx = cv2.minEnclosingCircle(contour)
            cx, cy = int(cx), int(cy)

            estimated_electrode_radius = radius_approx * 3.5

            if not (min_radius_pixel < estimated_electrode_radius < max_radius_pixel):
                continue

            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                if circularity < 0.70:  # Era 0.6
                    continue
            else:
                continue

            r_int = int(estimated_electrode_radius)
            x1 = max(0, cx - r_int)
            y1 = max(0, cy - r_int)
            x2 = min(image.shape[1], cx + r_int)
            y2 = min(image.shape[0], cy + r_int)

            roi = image[y1:y2, x1:x2]
            roi_hsv = hsv[y1:y2, x1:x2]

            if roi.size == 0:
                continue

            has_blue_border = self._check_blue_border(roi_hsv, roi)
            center_uniformity = self._check_center_uniformity(gray[y1:y2, x1:x2])

            if has_blue_border and center_uniformity > 0.7:  # Era 0.6
                confidence = (circularity * 0.3 +
                              center_uniformity * 0.4 +
                              (0.3 if has_blue_border else 0))

                detected.append({
                    'position': (cx, cy),
                    'bbox': (x1, y1, x2 - x1, y2 - y1),
                    'color': 'electrode',
                    'area': (x2 - x1) * (y2 - y1),
                    'circularity': circularity,
                    'aspect_ratio': 1.0,
                    'confidence': confidence,
                    'method': 'structured',
                    'radius': r_int
                })

        return detected

    def _check_blue_border(self, roi_hsv: np.ndarray, roi_bgr: np.ndarray) -> bool:
        """Check if there's a blue or dark border around the white center"""
        if roi_hsv.size == 0:
            return False

        h, w = roi_hsv.shape[:2]

        center = (w // 2, h // 2)
        outer_radius = min(w, h) // 2
        inner_radius = int(outer_radius * 0.6)

        border_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(border_mask, center, outer_radius, 255, -1)
        cv2.circle(border_mask, center, inner_radius, 0, -1)

        border_pixels = roi_hsv[border_mask > 0]

        if len(border_pixels) == 0:
            return False

        blue_mask = ((border_pixels[:, 0] >= 90) & (border_pixels[:, 0] <= 130) &
                     (border_pixels[:, 1] > 30) & (border_pixels[:, 2] > 30))

        blue_ratio = np.sum(blue_mask) / len(border_pixels)

        dark_mask = border_pixels[:, 2] < 100
        dark_ratio = np.sum(dark_mask) / len(border_pixels)

        return blue_ratio > 0.15 or dark_ratio > 0.2

    def _check_center_uniformity(self, roi_gray: np.ndarray) -> float:
        """Check if the center region is uniform"""
        if roi_gray.size == 0:
            return 0.0

        std_dev = np.std(roi_gray)
        uniformity = max(0, 1.0 - (std_dev / 50.0))

        return uniformity

    def _validate_electrodes(self, image: np.ndarray, electrodes: List[Dict]) -> List[Dict]:
        """Additional validation to filter out false positives"""
        validated = []

        for electrode in electrodes:
            x, y, w, h = electrode['bbox']

            roi = image[y:y + h, x:x + w]

            if roi.size == 0:
                continue

            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray_roi, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size

            if edge_density > 0.12:  # ERA 0.15 - ora più restrittivo
                continue

            hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

            center_h = h // 2
            center_w = w // 2
            center_region = hsv_roi[center_h - 5:center_h + 5, center_w - 5:center_w + 5]

            if center_region.size > 0:
                avg_value = np.mean(center_region[:, :, 2])
                if avg_value < 190:  # ERA 180 - ora più restrittivo
                    continue

            validated.append(electrode)

        return validated

    def _detect_colored_sensors(self, image: np.ndarray, min_area: int, max_area: int,
                                min_circularity: float, max_sensors: int) -> List[Dict]:
        """Original method for detecting colored sensors"""
        print("   🎨 Modalità rilevamento sensori colorati")

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        sensors = []

        color_to_search = ['red', 'blue', 'green', 'yellow']

        for color_name in color_to_search:
            if color_name not in self.color_ranges:
                continue

            for range_config in self.color_ranges[color_name]:
                lower = np.array(range_config['lower'])
                upper = np.array(range_config['upper'])

                mask = cv2.inRange(hsv, lower, upper)

                kernel = np.ones((3, 3), np.uint8)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

                found_sensors = self._find_sensors_in_mask(
                    mask, color_name, image.shape,
                    min_area, max_area, min_circularity
                )

                if found_sensors:
                    sensors.extend(found_sensors)

        sensors = self._remove_duplicate_sensors(sensors)

        if len(sensors) > max_sensors:
            sensors = sorted(sensors, key=lambda s: s['confidence'], reverse=True)[:max_sensors]

        return sensors

    def _find_sensors_in_mask(self, mask: np.ndarray, color: str, image_shape: Tuple,
                              min_area: int, max_area: int, min_circularity: float) -> List[Dict]:
        """Find sensors in binary mask"""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detected_sensors = []
        h, w = image_shape[:2]

        for contour in contours:
            area = cv2.contourArea(contour)

            if not (min_area < area < max_area):
                continue

            x, y, w_rect, h_rect = cv2.boundingRect(contour)

            aspect_ratio = float(w_rect) / h_rect if h_rect > 0 else 0
            if aspect_ratio < 0.5 or aspect_ratio > 2.0:
                continue

            center_x = x + w_rect // 2
            center_y = y + h_rect // 2

            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
            else:
                continue

            if circularity < min_circularity:
                continue

            margin = 20
            if (x < margin or y < margin or
                    x + w_rect > w - margin or y + h_rect > h - margin):
                continue

            area_score = min(area / max_area, 1.0)
            circularity_score = circularity
            confidence = (area_score * 0.4 + circularity_score * 0.6)

            detected_sensors.append({
                'position': (center_x, center_y),
                'bbox': (x, y, w_rect, h_rect),
                'color': color,
                'area': area,
                'circularity': circularity,
                'aspect_ratio': aspect_ratio,
                'confidence': confidence,
                'method': 'color'
            })

        return detected_sensors

    def _remove_duplicate_sensors(self, sensors: List[Dict], distance_threshold: int = 30) -> List[Dict]:
        """Remove duplicate sensor detections based on distance"""
        if len(sensors) <= 1:
            return sensors

        sensors = sorted(sensors, key=lambda s: s['confidence'], reverse=True)

        filtered_sensors = []

        for sensor in sensors:
            is_duplicate = False
            for existing in filtered_sensors:
                dist = np.linalg.norm(
                    np.array(sensor['position']) - np.array(existing['position'])
                )
                if dist < distance_threshold:
                    is_duplicate = True
                    break

            if not is_duplicate:
                filtered_sensors.append(sensor)

        return filtered_sensors

    def draw_sensors_on_image(self, image: np.ndarray, sensors: List[Dict]) -> np.ndarray:
        for s in sensors:
            cx, cy = s['position']
            radius = s.get('radius', 30)
            # Cerchio verde attorno all’elettrodo
            cv2.circle(image, (cx, cy), radius, (0, 255, 0), 3)
            # Punto centrale bianco
            cv2.circle(image, (cx, cy), 5, (255, 255, 255), -1)
            # Testo con confidenza
            cv2.putText(image, f"{s['confidence']:.2f}", (cx + 10, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        return image

    def _get_color_rgb(self, color_name: str) -> Tuple[int, int, int]:
        """Get RGB color tuple from color name"""
        colors = {
            'red': (0, 0, 255),
            'green': (0, 255, 0),
            'blue': (255, 0, 0),
            'yellow': (0, 255, 255),
            'black': (128, 128, 128),
            'white': (255, 255, 255),
            'electrode': (255, 0, 255)  # VIOLA invece di verde
        }
        return colors.get(color_name, (255, 255, 255))