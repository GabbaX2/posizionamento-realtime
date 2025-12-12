import cv2
import numpy as np


def check_sensor_color(roi: np.ndarray, threshold: float = 0.05) -> bool:
    """
    Controlla se la ROI contiene i colori specifici del sensore.
    threshold: percentuale minima di pixel (0.05 = 5%)

    FIX: Gestisce automaticamente immagini Grayscale o BGRA convertendole in BGR.
    """
    try:
        if roi is None or roi.size == 0:
            return False

        # --- FIX ROBUSTEZZA CANALI ---
        # Se l'immagine è in scala di grigi (2 dimensioni: H, W), convertila in BGR
        if len(roi.shape) == 2:
            roi = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)

        # Se l'immagine ha 4 canali (BGRA), convertila in BGR
        elif roi.shape[2] == 4:
            roi = cv2.cvtColor(roi, cv2.COLOR_BGRA2BGR)

        # Ora siamo sicuri che roi sia BGR (3 canali)
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # 1. AZZURRO (Gel)
        lower_cyan = np.array([75, 20, 70])
        upper_cyan = np.array([145, 255, 255])
        mask_cyan = cv2.inRange(hsv, lower_cyan, upper_cyan)

        # 2. GRIGIO (Bottone)
        lower_grey = np.array([0, 0, 50])
        upper_grey = np.array([180, 50, 220])
        mask_grey = cv2.inRange(hsv, lower_grey, upper_grey)

        # 3. BIANCO (Bordo)
        lower_white = np.array([0, 0, 160])
        upper_white = np.array([180, 60, 255])
        mask_white = cv2.inRange(hsv, lower_white, upper_white)

        combined_mask = cv2.bitwise_or(mask_cyan, mask_grey)
        combined_mask = cv2.bitwise_or(combined_mask, mask_white)

        non_zero = cv2.countNonZero(combined_mask)
        total_pixels = roi.shape[0] * roi.shape[1]

        return (non_zero / total_pixels) > threshold

    except Exception as e:
        # È buona norma loggare l'errore se possibile, o ritornare False
        print(f"Error in check_sensor_color: {e}")
        return False