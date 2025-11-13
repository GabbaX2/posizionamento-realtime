from rembg import remove
from PIL import Image
import io
import base64
import numpy as np
import cv2
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------- Helpers (Invariati) ----------
def _ensure_rgb(pil_img):
    if pil_img.mode in ("RGBA", "LA"):
        return pil_img.convert("RGB")
    if pil_img.mode != "RGB":
        return pil_img.convert("RGB")
    return pil_img


def _auto_contrast_and_denoise_cv(img_bgr):
    img = img_bgr.copy()
    img = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
    img = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    return img


def _resize_for_speed(pil_img, max_dim=1024):
    w, h = pil_img.size
    max_current = max(w, h)
    if max_current <= max_dim:
        return pil_img, 1.0
    scale = max_dim / float(max_current)
    new_size = (int(w * scale), int(h * scale))
    return pil_img.resize(new_size, Image.LANCZOS), scale


# ---------- GrabCut fallback (Invariato) ----------
def grabcut_fallback_bgra(input_bgr, iter_count=5):
    h, w = input_bgr.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    rect = (int(w * 0.05), int(h * 0.05), int(w * 0.9), int(h * 0.9))
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(input_bgr, mask, rect, bgdModel, fgdModel, iter_count, cv2.GC_INIT_WITH_RECT)
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        bgr = input_bgr.copy() * mask2[:, :, None]
        alpha = (mask2 * 255).astype('uint8')
        bgra = cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
        bgra[:, :, 3] = alpha
        return bgra
    except Exception:
        logger.exception("GrabCut fallback failed")
        return None


# ---------- Main (Modificato per Pulizia Maschera) ----------
def remove_background_from_pil(pil_image, use_preprocess=True, resize_max=1024, grabcut_if_failed=True):
    """
    Input: PIL Image
    Output: (PIL Image in RGBA, metodo_usato) oppure (None, None)
    """
    try:
        pil_image_rgb = _ensure_rgb(pil_image)
        pil_small, scale = _resize_for_speed(pil_image_rgb, max_dim=resize_max)

        out_small = None
        method_used = None

        # ==========================================================
        # STEP 1: Rimuovi sfondo con un modello di segmentazione (bordi netti)
        # ==========================================================
        try:
            # Usiamo 'u2net' (o 'isnet-general-use') che è fatto per la segmentazione,
            # non 'alpha_matting' che crea bordi sfumati/blurred.
            out_small = remove(pil_small, model_name="u2net")
            method_used = "rembg-u2net"

        except Exception:
            logger.exception("rembg.remove ha sollevato un'eccezione")
            out_small = None
        # ==========================================================
        # Fine STEP 1 Modificato
        # ==========================================================

        if out_small is not None:
            if out_small.mode != "RGBA":
                out_small = out_small.convert("RGBA")
            alpha_check = np.array(out_small.split()[-1])
            if np.count_nonzero(alpha_check > 10) < 10:
                logger.info("rembg ha prodotto alpha vuoto, passo al fallback")
                out_small = None

        # STEP 2: fallback GrabCut (Invariato)
        # (Questo verrà usato solo se rembg fallisce completamente)
        if grabcut_if_failed and out_small is None:
            logger.warning("rembg ha fallito, tentativo con GrabCut...")
            img_bgr = cv2.cvtColor(np.array(pil_small), cv2.COLOR_RGB2BGR)
            if use_preprocess:
                img_bgr = _auto_contrast_and_denoise_cv(img_bgr)
            bgra_small = grabcut_fallback_bgra(img_bgr)
            if bgra_small is not None:
                out_small = Image.fromarray(cv2.cvtColor(bgra_small, cv2.COLOR_BGRA2RGBA))
                method_used = "grabcut"

        if out_small is None:
            logger.error("Tutti i metodi di rimozione sfondo hanno fallito.")
            return None, None

        # ==========================================================
        # STEP 3: Pulizia aggressiva della Maschera
        # ==========================================================
        logger.info(f"Pulizia maschera prodotta da '{method_used}'...")

        # Estrai la maschera alpha (potrebbe essere sfumata o sporca)
        alpha = np.array(out_small.split()[-1])

        # 1. Binarizzazione: Trasforma i bordi sfumati (grigi) in bordi netti (bianco o nero)
        _, alpha_binary = cv2.threshold(alpha, 127, 255, cv2.THRESH_BINARY)

        # 2. Pulizia Rumore: Rimuovi piccoli "detriti"
        kernel_open = np.ones((3, 3), np.uint8)
        alpha_cleaned = cv2.morphologyEx(alpha_binary, cv2.MORPH_OPEN, kernel_open, iterations=2)

        # 3. Chiusura Buchi: Riempi piccoli buchi *dentro* la gamba
        kernel_close = np.ones((5, 5), np.uint8)
        alpha_cleaned = cv2.morphologyEx(alpha_cleaned, cv2.MORPH_CLOSE, kernel_close, iterations=3)

        # 4. Trova il Contorno Più Grande: Isola la gamba
        contours, _ = cv2.findContours(alpha_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        final_alpha = np.zeros_like(alpha_cleaned)  # Crea una nuova maschera vuota

        if contours:
            main_contour = max(contours, key=cv2.contourArea)
            # 5. Crea la Maschera Perfetta: Disegna solo la gamba (piena)
            cv2.drawContours(final_alpha, [main_contour], -1, (255), cv2.FILLED)

        # Applica la nuova maschera pulita all'immagine
        out_small.putalpha(Image.fromarray(final_alpha))
        # ==========================================================
        # Fine STEP 3 Modificato
        # ==========================================================

        # STEP 4: resize back (Invariato)

        if scale != 1.0 and out_small is not None:
            orig_w, orig_h = pil_image.size
            out_full = out_small.resize((orig_w, orig_h), Image.LANCZOS)
        else:
            out_full = out_small

        # Rimuoviamo il vecchio post-processing (Step 4), non ci serve più
        logger.info(f"Rimozione sfondo completata con successo usando '{method_used}' (e pulizia maschera).")
        return out_full.convert("RGBA"), method_used

    except Exception:
        logger.exception("Errore in remove_background_from_pil")
        return None, None