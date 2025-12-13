import cv2
import numpy as np
import base64
from typing import Dict, List
from PIL import Image

from utils.background_removal import remove_background_from_pil


class ReferenceProcessor:
    def __init__(self):
        self.STEP_SIZE = 2
        # Colore Arancione (BGR)
        self.DRAWING_COLOR = (0, 165, 255)

        # --- CONFIGURAZIONE UI AGGIORNATA ---
        self.FONT = cv2.FONT_HERSHEY_SIMPLEX
        self.FONT_SCALE = 1.3  # Aumentato (era 0.8)
        self.FONT_THICKNESS = 3  # Aumentato (era 2)
        self.FIXED_RADIUS = 30  # Dimensione fissa visuale per tutti i cerchi
        # ------------------------------------

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

        # Rimozione sfondo
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

        # --- PREPARAZIONE ---
        comp_image = np.full((h, w, 3), 127, dtype=np.uint8)
        mask_leg = alpha > 0
        comp_image[mask_leg] = img_bgra[mask_leg, :3]
        gray = cv2.cvtColor(comp_image, cv2.COLOR_BGR2GRAY)

        # Disegno contorno gamba
        _, thresh = cv2.threshold(alpha, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        drawing_mask = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.drawContours(drawing_mask, contours, -1, self.DRAWING_COLOR, 2)

        # --- HOUGH ---
        gray_blur = cv2.medianBlur(gray, 11)
        circles = cv2.HoughCircles(
            gray_blur, cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=60,
            param1=50,
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

        valid_sensors = self._remove_duplicates_strict(candidates)
        valid_sensors.sort(key=lambda v: (v[1], v[0]))

        steps_config = {}

        # --- DISEGNO UI ---
        for i in range(0, len(valid_sensors), self.STEP_SIZE):
            chunk = valid_sensors[i: i + self.STEP_SIZE]
            step_idx = (i // self.STEP_SIZE) + 1

            if not chunk: continue

            step_list = []

            # Liste per calcolare il centro del GRUPPO
            group_x_coords = []
            group_y_coords = []

            for s in chunk:
                cx, cy, detected_r = s

                # 1. Disegno cerchio con RAGGIO FISSO (ignoriamo detected_r per il disegno)
                cv2.circle(drawing_mask, (cx, cy), self.FIXED_RADIUS, self.DRAWING_COLOR, 3)

                # 2. Centro pieno
                cv2.circle(drawing_mask, (cx, cy), 5, self.DRAWING_COLOR, -1)

                # Salviamo i dati reali per la config (qui manteniamo il raggio rilevato o fisso, a tua scelta)
                # Se vuoi coerenza totale anche nei dati, usa self.FIXED_RADIUS anche qui.
                step_list.append({'id': i + 1, 'x': cx, 'y': cy, 'r': self.FIXED_RADIUS})

                group_x_coords.append(cx)
                group_y_coords.append(cy)

            steps_config[str(step_idx)] = step_list

            # --- CALCOLO POSIZIONE TESTO ---
            # Calcoliamo il centro geometrico del gruppo
            center_group_x = int(sum(group_x_coords) / len(group_x_coords))
            # Prendiamo la Y più alta (minore) del gruppo per mettere il testo sopra a tutto
            top_group_y = min(group_y_coords)

            label_text = f"Gruppo {step_idx}"

            # Calcolo dimensioni testo
            (text_w, text_h), baseline = cv2.getTextSize(
                label_text, self.FONT, self.FONT_SCALE, self.FONT_THICKNESS
            )

            # Centriamo il testo rispetto alla X media del gruppo
            text_x = center_group_x - (text_w // 2)
            # Posizioniamo sopra il cerchio più alto, considerando il raggio fisso e un margine
            text_y = top_group_y - self.FIXED_RADIUS - 15

            # Protezione bordi (evita testo fuori schermo)
            text_x = max(10, min(w - text_w - 10, text_x))
            text_y = max(text_h + 10, text_y)

            cv2.putText(
                drawing_mask,
                label_text,
                (text_x, text_y),
                self.FONT,
                self.FONT_SCALE,
                self.DRAWING_COLOR,
                self.FONT_THICKNESS,
                cv2.LINE_AA
            )

        return drawing_mask, steps_config

    def _remove_duplicates_strict(self, circles):
        if not circles: return []
        circles = sorted(circles, key=lambda x: x[2], reverse=True)
        kept = []
        for current in circles:
            cx, cy, cr = current
            is_duplicate = False
            for kept_c in kept:
                kx, ky, kr = kept_c
                dist = np.sqrt((cx - kx) ** 2 + (cy - ky) ** 2)
                if dist < 50 or dist < kr:
                    is_duplicate = True
                    break
            if not is_duplicate:
                kept.append(current)
        return kept

    def _check_color_signature(self, roi):
        if roi.size == 0: return False
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        lower_white = np.array([0, 0, 90])
        upper_white = np.array([180, 60, 255])
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