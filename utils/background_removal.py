from rembg import remove
from PIL import Image
import io
import base64
import numpy as np
import cv2


def remove_background(image_data):
    """
    Rimuove lo sfondo da un'immagine in formato base64
    
    Args:
        image_data: immagine in formato base64 (con o senza prefisso data:image)
    
    Returns:
        str: immagine con sfondo rimosso in formato base64 (solo base64, senza prefisso)
        None: in caso di errore
    """
    try:
        # Se l'immagine è in base64 con prefisso, rimuovilo
        if isinstance(image_data, str):
            if ',' in image_data:
                image_data = image_data.split(',')[1]
            image_bytes = base64.b64decode(image_data)
        else:
            image_bytes = image_data
        
        # Apri l'immagine
        input_image = Image.open(io.BytesIO(image_bytes))
        
        # Rimuovi lo sfondo
        output_image = remove(input_image)
        
        # Converti in base64
        buffered = io.BytesIO()
        output_image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()
        
        return img_base64
    except Exception as e:
        print(f"[ERROR] Errore nella rimozione dello sfondo: {e}")
        import traceback
        traceback.print_exc()
        return None


def remove_background_from_file(file):
    """
    Rimuove lo sfondo da un file caricato (Flask request.files)
    
    Args:
        file: file object da Flask request.files
    
    Returns:
        str: immagine con sfondo rimosso in formato base64 (solo base64, senza prefisso)
        None: in caso di errore
    """
    try:
        # Leggi il file
        image_bytes = file.read()
        
        # Apri l'immagine
        input_image = Image.open(io.BytesIO(image_bytes))
        
        # Rimuovi lo sfondo
        output_image = remove(input_image)
        
        # Converti in base64
        buffered = io.BytesIO()
        output_image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()
        
        return img_base64
    except Exception as e:
        print(f"[ERROR] Errore nella rimozione dello sfondo dal file: {e}")
        import traceback
        traceback.print_exc()
        return None


def remove_background_from_cv2_image(image):
    """
    Rimuove lo sfondo da un'immagine OpenCV (numpy array)
    
    Args:
        image: immagine OpenCV (numpy array BGR)
    
    Returns:
        numpy.ndarray: immagine con sfondo rimosso (BGRA con alpha channel)
        None: in caso di errore
    """
    try:
        # Converti BGR a RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Converti in PIL Image
        pil_image = Image.fromarray(image_rgb)
        
        # Rimuovi lo sfondo
        output_image = remove(pil_image)
        
        # Converti di nuovo in OpenCV (BGRA)
        output_array = np.array(output_image)
        
        # Se l'immagine ha 4 canali (RGBA), convertila in BGRA per OpenCV
        if output_array.shape[2] == 4:
            output_bgra = cv2.cvtColor(output_array, cv2.COLOR_RGBA2BGRA)
            return output_bgra
        else:
            # Se per qualche motivo non ha alpha, aggiungi un canale alpha
            output_bgr = cv2.cvtColor(output_array, cv2.COLOR_RGB2BGR)
            return output_bgr
            
    except Exception as e:
        print(f"[ERROR] Errore nella rimozione dello sfondo da immagine CV2: {e}")
        import traceback
        traceback.print_exc()
        return None


def create_white_background_image(image_with_alpha):
    """
    Sostituisce lo sfondo trasparente con uno sfondo bianco
    Utile per la detection che funziona meglio con sfondi uniformi
    
    Args:
        image_with_alpha: immagine BGRA con canale alpha
    
    Returns:
        numpy.ndarray: immagine BGR con sfondo bianco
    """
    try:
        # Crea uno sfondo bianco
        white_bg = np.ones_like(image_with_alpha[:, :, :3]) * 255
        
        # Estrai il canale alpha e normalizzalo
        alpha = image_with_alpha[:, :, 3:4] / 255.0
        
        # Blend l'immagine con lo sfondo bianco usando il canale alpha
        result = (image_with_alpha[:, :, :3] * alpha + white_bg * (1 - alpha)).astype(np.uint8)
        
        return result
    except Exception as e:
        print(f"[ERROR] Errore nella creazione dello sfondo bianco: {e}")
        return image_with_alpha[:, :, :3] if image_with_alpha.shape[2] == 4 else image_with_alpha


def create_colored_background_image(image_with_alpha, color=(255, 255, 255)):
    """
    Sostituisce lo sfondo trasparente con un colore specifico
    
    Args:
        image_with_alpha: immagine BGRA con canale alpha
        color: tupla BGR del colore di sfondo (default: bianco)
    
    Returns:
        numpy.ndarray: immagine BGR con sfondo colorato
    """
    try:
        # Crea uno sfondo del colore specificato
        colored_bg = np.full_like(image_with_alpha[:, :, :3], color, dtype=np.uint8)
        
        # Estrai il canale alpha e normalizzalo
        alpha = image_with_alpha[:, :, 3:4] / 255.0
        
        # Blend l'immagine con lo sfondo colorato usando il canale alpha
        result = (image_with_alpha[:, :, :3] * alpha + colored_bg * (1 - alpha)).astype(np.uint8)
        
        return result
    except Exception as e:
        print(f"[ERROR] Errore nella creazione dello sfondo colorato: {e}")
        return image_with_alpha[:, :, :3] if image_with_alpha.shape[2] == 4 else image_with_alpha