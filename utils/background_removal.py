from rembg import remove
from PIL import Image
import numpy as np
import cv2
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------- Helpers ----------
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


# ---------- GrabCut fallback ----------
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


# ---------- Main ----------
def remove_background_from_pil(pil_image, use_preprocess=True, resize_max=1024, grabcut_if_failed=True):
    """
    Input: PIL Image
    Output: (PIL Image in RGBA, metodo_usato) oppure (None, None)
    """
    try:
        pil_image = _ensure_rgb(pil_image)
        pil_small, scale = _resize_for_speed(pil_image, max_dim=resize_max)

        out_small = None
        method_used = None

        # STEP 1: rembg con modello più robusto
        try:
            out_small = remove(pil_small, model_name="isnet-general-use")
            method_used = "rembg-isnet"
        except Exception:
            logger.exception("rembg.remove ha sollevato un'eccezione")
            out_small = None

        if out_small is not None:
            if out_small.mode != "RGBA":
                out_small = out_small.convert("RGBA")
            alpha = np.array(out_small.split()[-1])
            if np.count_nonzero(alpha > 10) < 10:
                logger.info("rembg ha prodotto alpha vuoto, passo al fallback")
                out_small = None

        # STEP 2: fallback GrabCut
        grabcut_mask = None
        if grabcut_if_failed or out_small is None:
            img_bgr = cv2.cvtColor(np.array(pil_small), cv2.COLOR_RGB2BGR)
            if use_preprocess:
                img_bgr = _auto_contrast_and_denoise_cv(img_bgr)
            bgra_small = grabcut_fallback_bgra(img_bgr)
            if bgra_small is not None:
                grabcut_img = Image.fromarray(cv2.cvtColor(bgra_small, cv2.COLOR_BGRA2RGBA))
                grabcut_mask = np.array(grabcut_img.split()[-1])
                if out_small is None:
                    out_small = grabcut_img
                    method_used = "grabcut"
                else:
                    alpha_rembg = np.array(out_small.split()[-1])
                    fused_alpha = cv2.bitwise_or(alpha_rembg, grabcut_mask)
                    out_small.putalpha(Image.fromarray(fused_alpha))
                    method_used = "rembg+grabcut"

        # STEP 3: resize back
        if scale != 1.0 and out_small is not None:
            orig_w, orig_h = pil_image.size
            out_full = out_small.resize((orig_w, orig_h), Image.LANCZOS)
        else:
            out_full = out_small

        if out_full is None:
            return None, None

        # STEP 4: post-processing maschera
        alpha = np.array(out_full.split()[-1])
        kernel = np.ones((5, 5), np.uint8)
        alpha = cv2.morphologyEx(alpha, cv2.MORPH_CLOSE, kernel, iterations=2)
        alpha = cv2.dilate(alpha, kernel, iterations=1)
        out_full.putalpha(Image.fromarray(alpha))

        return out_full.convert("RGBA"), method_used
    except Exception:
        logger.exception("Errore in remove_background_from_pil")
        return None, None
