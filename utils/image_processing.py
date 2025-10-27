import os
import cv2
import numpy as np
import base64
from typing import Tuple, Optional
from flask import current_app


def allowed_file(filename: str) -> bool:
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in current_app.config['ALLOWED_EXTENSIONS']


def save_uploaded_file(file, limb_type: str, session_id: str = 'default') -> Optional[str]:
    try:
        if file and allowed_file(file.filename):
            filename = f"reference_{limb_type}_{session_id}.jpg"
            filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)

            file.save(filepath)
            return filepath
    except Exception as e:
        print(f'Error saving file: {e}')
    return None


def encode_image_to_base64(image: np.ndarray) -> str:
    try:
        _, buffer = cv2.imencode('.jpg', image)
        return base64.b64encode(buffer).decode('utf-8')
    except Exception as e:
        print(f'Error encoding image: {e}')
        return ""


def resize_image(image: np.ndarray, max_width: int = 800, max_height: int = 600) -> np.ndarray:
    h, w = image.shape[:2]

    if w > max_width or h > max_height:
        scale = min(max_width / w, max_height / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        return cv2.resize(image, (new_w, new_h))

    return image


def preprocess_image(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Apply slight sharpening
    kernel = np.array([[-1, -1, -1],
                       [-1, 9, -1],
                       [-1, -1, -1]])
    sharpened = cv2.filter2D(image, -1, kernel)

    return sharpened


def extract_roi(image: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
    """Extract region of interest from image"""
    x, y, w, h = bbox
    return image[y:y+h, x:x+w]