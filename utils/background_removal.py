import io
import logging
import os
import numpy as np
import cv2
import requests
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

REMOVE_BG_API_URL = "https://api.remove.bg/v1.0/removebg"


def _call_remove_bg_api(pil_img_rgb, api_key):
    """Chiama l'API commerciale remove.bg."""
    try:
        img_io = io.BytesIO()
        pil_img_rgb.save(img_io, format="PNG")
        img_io.seek(0)

        files = {'image_file': ('image.png', img_io, 'image/png')}
        data = {'size': 'auto'}
        headers = {'X-Api-Key': api_key}

        logger.info("Chiamata API remove.bg in corso...")
        response = requests.post(
            REMOVE_BG_API_URL,
            files=files,
            data=data,
            headers=headers,
            timeout=30
        )

        if response.status_code == requests.codes.ok:
            result_img_io = io.BytesIO(response.content)
            return Image.open(result_img_io).convert("RGBA")
        else:
            logger.error(f"Errore API {response.status_code}: {response.text}")
            return None

    except requests.exceptions.RequestException as e:
        logger.warning(f"Eccezione connessione API: {e}")
        return None
    except Exception:
        logger.exception("Errore generico chiamata API")
        return None


def _ensure_rgb(pil_img):
    """Converte immagine PIL in RGB."""
    if pil_img.mode in ("RGBA", "LA") or pil_img.mode != "RGB":
        return pil_img.convert("RGB")
    return pil_img


def _resize_for_speed(pil_img, max_dim=1024):
    """Ridimensiona mantenendo aspect ratio."""
    w, h = pil_img.size
    max_current = max(w, h)
    if max_current <= max_dim:
        return pil_img, 1.0
    scale = max_dim / float(max_current)
    new_size = (int(w * scale), int(h * scale))
    return pil_img.resize(new_size, Image.LANCZOS), scale


def grabcut_fallback_bgra(input_bgr, iter_count=5):
    """Fallback locale con GrabCut."""
    h, w = input_bgr.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    rect = (1, 1, w - 2, h - 2)

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


def remove_background_from_pil(pil_image, api_key=None, resize_max=1024, grabcut_if_failed=True):
    """
    Input: Immagine PIL
    Output: (Immagine PIL RGBA, metodo_usato)
    """
    try:
        if api_key is None:
            api_key = os.getenv("REMOVE_BG_API_KEY")

        pil_image_rgb = _ensure_rgb(pil_image)
        pil_small, scale = _resize_for_speed(pil_image_rgb, max_dim=resize_max)

        out_small = None
        method_used = None

        # 1. Tentativo API
        if api_key:
            try:
                out_small = _call_remove_bg_api(pil_small, api_key)
                if out_small:
                    method_used = "api-remove-bg-commercial"
            except Exception:
                logger.exception("Errore chiamata API")
                out_small = None
        else:
            logger.info("Nessuna API Key trovata, salto al fallback.")

        # Validazione risultato
        if out_small is not None:
            if out_small.mode != "RGBA":
                out_small = out_small.convert("RGBA")
            alpha_check = np.array(out_small.split()[-1])
            if np.count_nonzero(alpha_check > 10) < 10:
                out_small = None
                method_used = None

        # 2. Tentativo Fallback
        if grabcut_if_failed and out_small is None:
            logger.warning("Avvio GrabCut locale.")
            img_bgr = cv2.cvtColor(np.array(pil_small), cv2.COLOR_RGB2BGR)
            bgra_small = grabcut_fallback_bgra(img_bgr)

            if bgra_small is not None:
                out_small = Image.fromarray(cv2.cvtColor(bgra_small, cv2.COLOR_BGRA2RGBA))
                method_used = "grabcut-fallback"

        if out_small is None:
            return None, None

        # Post-processing (Morphological Cleaning)
        alpha = np.array(out_small.split()[-1])
        _, alpha_binary = cv2.threshold(alpha, 127, 255, cv2.THRESH_BINARY)

        kernel_close = np.ones((7, 7), np.uint8)
        alpha_closed = cv2.morphologyEx(alpha_binary, cv2.MORPH_CLOSE, kernel_close, iterations=2)

        kernel_open = np.ones((3, 3), np.uint8)
        alpha_cleaned = cv2.morphologyEx(alpha_closed, cv2.MORPH_OPEN, kernel_open, iterations=1)

        out_small.putalpha(Image.fromarray(alpha_cleaned))

        # Ripristino dimensioni
        if scale != 1.0 and out_small is not None:
            orig_w, orig_h = pil_image.size
            out_full = out_small.resize((orig_w, orig_h), Image.LANCZOS)
        else:
            out_full = out_small

        return out_full.convert("RGBA"), method_used

    except Exception:
        logger.exception("Errore critico in remove_background_from_pil")
        return None, None