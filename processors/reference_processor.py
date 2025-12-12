import cv2
import numpy as np
import base64
from typing import Dict, List
from PIL import Image

# Assicurati che i path siano corretti per il tuo progetto
from utils.background_removal import remove_background_from_pil
from utils.geometry_utils import calculate_distance


class ReferenceProcessor:
    def __init__(self):
        self.STEP_SIZE = 2
        self.DRAWING_COLOR = (0, 165, 255)  # Arancione

    def process(self, file_stream) -> Dict:
        # 1. Caricamento
        file_stream.seek(0)
        nparr = np.frombuffer(file_stream.read(), np.uint8)
        img_input = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)

        if img_input is None:
            return {'success': False, 'error': 'Impossibile decodificare immagine'}

        # Normalizzazione BGRA
        if len(img_input.shape) == 2:
            img_bgra = cv2.cvtColor(img_input, cv2.COLOR_GRAY2BGRA)
        elif img_input.shape[2] == 3:
            b, g, r = cv2.split(img_input)
            alpha = np.ones_like(b) * 255
            img_bgra = cv2.merge((b, g, r, alpha))
        else:
            img_bgra = img_input

        # Rimozione sfondo (solo se immagine opaca)
        if np.min(img_bgra[:, :, 3]) == 255:
            pil_img = Image.fromarray(cv2.cvtColor(img_bgra, cv2.COLOR_BGRA2RGBA))
            pil_out, _ = remove_background_from_pil(pil_img)
            if pil_out:
                img_bgra = cv2.cvtColor(np.array(pil_out), cv2.COLOR_RGBA2BGRA)

        drawing_mask, steps_config = self._analyze_image(img_bgra)

        return {
            'success': True,
            'reference_clean_b64': self._to_b64(img_bgra[:, :, :3]),
            'drawing_mask_b64': self._to_b64(drawing_mask),
            'steps_config': steps_config,
            'dims': (img_bgra.shape[0], img_bgra.shape[1])
        }

    def _analyze_image(self, img_bgra):
        h, w = img_bgra.shape[:2]
        alpha = img_bgra[:, :, 3]

        # --- PREPARAZIONE SFONDO GRIGIO ---
        comp_image = np.full((h, w, 3), 127, dtype=np.uint8)
        mask_leg = alpha > 0
        comp_image[mask_leg] = img_bgra[mask_leg, :3]
        gray = cv2.cvtColor(comp_image, cv2.COLOR_BGR2GRAY)

        # Disegno contorno gamba per UI
        _, thresh = cv2.threshold(alpha, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        drawing_mask = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.drawContours(drawing_mask, contours, -1, self.DRAWING_COLOR, 3)

        # --- CONFIGURAZIONE HOUGH MOLTO RIGIDA ---
        # Blur alto per fondere vite metallica e adesivo in un'unica "macchia" tonda
        gray_blur = cv2.medianBlur(gray, 11)

        circles = cv2.HoughCircles(
            gray_blur, cv2.HOUGH_GRADIENT,
            dp=1.2,
            # DISTANZA MINIMA: 60px. Impedisce fisicamente di trovare due cerchi vicini.
            # Se i sensori reali distano meno di 60px, abbassa questo valore a 40.
            minDist=60,
            param1=50,
            # SOGLIA CERTEZZA: Alzata a 45. Scarta cerchi deboli.
            param2=45,
            minRadius=15,
            maxRadius=65
        )

        img_rgb = img_bgra[:, :, :3]
        candidates = []

        if circles is not None:
            circles = np.uint16(np.around(circles))
            for (x, y, r) in circles[0, :]:
                if x >= w or y >= h: continue
                if alpha[int(y), int(x)] == 0: continue

                roi = img_rgb[max(0, y - r):min(h, y + r), max(0, x - r):min(w, x + r)]
                if self._check_color_signature(roi):
                    candidates.append((int(x), int(y), int(r)))

        # --- FILTRO DUPLICATI FINALE (Sicurezza Extra) ---
        valid_sensors = self._remove_duplicates_strict(candidates)

        # Ordinamento: Y (alto -> basso), poi X
        valid_sensors.sort(key=lambda v: (v[1], v[0]))

        steps_config = {}
        for i in range(0, len(valid_sensors), self.STEP_SIZE):
            chunk = valid_sensors[i: i + self.STEP_SIZE]
            step_idx = (i // self.STEP_SIZE) + 1
            step_list = []
            for s in chunk:
                # Disegno cerchio esterno
                cv2.circle(drawing_mask, (s[0], s[1]), s[2], self.DRAWING_COLOR, 2)
                # Disegno centro pieno per puntamento preciso
                cv2.circle(drawing_mask, (s[0], s[1]), 3, self.DRAWING_COLOR, -1)

                step_list.append({'id': i + 1, 'x': s[0], 'y': s[1], 'r': s[2]})

            steps_config[str(step_idx)] = step_list

        return drawing_mask, steps_config

    def _remove_duplicates_strict(self, circles):
        """
        Mantiene solo il cerchio più grande in ogni gruppo di sovrapposizione.
        Se due cerchi distano meno di 50px, vince il più grande.
        """
        if not circles: return []

        # 1. Ordina per Raggio (dal più grande al più piccolo)
        # Assumiamo che il cerchio più grande sia l'adesivo esterno (quello che vogliamo vedere)
        circles = sorted(circles, key=lambda x: x[2], reverse=True)
        kept = []

        for current in circles:
            cx, cy, cr = current
            is_duplicate = False

            for kept_c in kept:
                kx, ky, kr = kept_c

                # Distanza tra i centri
                dist = np.sqrt((cx - kx) ** 2 + (cy - ky) ** 2)

                # LOGICA DI ELIMINAZIONE:
                # 1. Se i centri sono vicini (meno di 50px) -> è lo stesso sensore -> SCARTA IL PIÙ PICCOLO (current)
                # 2. Se il cerchio attuale è fisicamente "dentro" quello già salvato (distanza < raggio del grande) -> SCARTA
                if dist < 50 or dist < kr:
                    is_duplicate = True
                    break

            if not is_duplicate:
                kept.append(current)

        return kept

    def _check_color_signature(self, roi):
        if roi.size == 0: return False
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # Bianco/Grigio (nucleo)
        lower_white = np.array([0, 0, 90])
        upper_white = np.array([180, 60, 255])

        # Blu (logo/bordo)
        lower_blue = np.array([80, 30, 30])
        upper_blue = np.array([140, 255, 255])

        mask_white = cv2.inRange(hsv, lower_white, upper_white)
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

        combined = cv2.bitwise_or(mask_white, mask_blue)
        ratio = cv2.countNonZero(combined) / (roi.shape[0] * roi.shape[1])

        return ratio > 0.15

    def _to_b64(self, img):
        _, buf = cv2.imencode('.png', img)
        return base64.b64encode(buf).decode()