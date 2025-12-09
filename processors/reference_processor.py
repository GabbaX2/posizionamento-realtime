import cv2
import numpy as np
import base64
from typing import Dict, Tuple, List
from PIL import Image

from utils.background_removal import remove_background_from_pil
from utils.sensor_utils import check_sensor_color
from utils.geometry_utils import calculate_distance


class ReferenceProcessor:
    def __init__(self):
        self.STEP_SIZE = 2

    def process(self, file_stream) -> Dict:
        """
        Input: File stream (dal form upload)
        Output: Dizionario con immagini b64 e configurazione sensori
        """

        # 1. Caricamento Immagine
        file_stream.seek(0)
        nparr = np.frombuffer(file_stream.read(), np.uint8)
        img_input = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)

        # 2. Rimozione sfondo
        # Se l'immagine non è già in PNG trasparente, puliscila
        if img_input.shape[2] == 3:
            pil_img = Image.fromarray(cv2.cvtColor(img_input, cv2.COLOR_BGR2RGB))
            pil_out, _ = remove_background_from_pil(pil_img)

            if pil_out:
                img_bgra = cv2.cvtColor(np.array(pil_out), cv2.COLOR_RGB2BGRA)
            else:
                # Fallback: aggiungi alpha opaco
                b, g, r = cv2.split(img_input)
                alpha = np.ones_like(b) * 255
                img_bgra = cv2.merge((b, g, r, alpha))
        else:
            img_bgra = img_input

        # 3. Creazione maschere e analisi
        drawing_mask, steps_config = self._analyze_image(img_bgra)

        # 4. Return dati puliti
        return {
            'success': True,
            'reference_clean_b64': self._to_b64(img_bgra[:, :, :3]),
            'drawing_mask_b64': self._to_b64(drawing_mask),  # Il "Disegno" bordi bianchi su nero
            'steps_config': steps_config,
            'dims': (img_bgra.shape[0], img_bgra.shape[1])
        }

    def _analyze_image(self, img_bgra):
        h, w = img_bgra.shape[:2]
        alpha = img_bgra[:, :, 3]

        # 1. Estrazione Contorni per allineamento
        _, thresh = cv2.threshold(alpha, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        drawing_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(drawing_mask, contours, -1, 255, 3) # Bordi bianchi spessore 3

        # 2. Rilevamento Sensori (Hough Circles + Colore)
        img_rgb = img_bgra[:, :, 3]
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.medianBlur(gray, 5)

        circles = cv2.HoughCircles(
            gray_blur, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50,
            param1=50, param2=30, minRadius=15, maxRadius=55
        )

        valid_sensors = []
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for (x, y, r) in circles[0, :]:
                roi = img_rgb[max(0, y-r):min(h, y+r), max(0, x-r):min(w, x+r)]
                if self._check_color_signature(roi):
                    valid_sensors.append((int(x), int(y), int(r)))

        valid_sensors = self._remove_overlapping_circles(valid_sensors, min_dist=50)

        # 3. Organizzazione in step
        valid_sensors.sort(key=lambda v: v[1]) # Ordina per Y
        steps_config = {}
        steps_idx = 1

        for i in range(0, len(valid_sensors), self.STEP_SIZE):
            chunk = valid_sensors[i : i + self.STEP_SIZE]
            step_list = []
            for sensor in chunk:
                # Disegna anche i sensori sulla maschera di riferimento (aiuta l'allineamento)
                cv2.circle(drawing_mask, (sensor[0], sensor[1]), sensor[2], 255, 2)

                step_list.append({
                    'id': i + len(step_list) + 1,
                    'x': sensor[0], 'y': sensor[1], 'r': sensor[2]
                })

            steps_config[str(steps_idx)] = step_list
            steps_idx += 1

        return drawing_mask, steps_config

    def _remove_overlapping_circles(self, circles, min_dist):
        if not circles: return []
        clean = []
        circles = sorted(circles, key=lambda x: x[2], reverse=True)

        for c in circles:
            x, y, r = c
            is_overlap = False
            for ex, ey, er in clean:
                if calculate_distance((x, y), (ex, ey)) < min_dist:
                    is_overlap = True
                    break
            if not is_overlap: clean.append(c)
        return clean

    def _to_b64(self, img):
        _, buf = cv2.imencode('.png', img)
        return base64.b64encode(buf).decode()
