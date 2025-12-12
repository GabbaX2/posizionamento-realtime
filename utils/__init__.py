# utils/__init__.py

# 1. Importa le funzioni geometriche (risolve il tuo errore su rotate_point)
from .geometry_utils import (
    calculate_distance,
    calculate_angle,
    rotate_point,
    calculate_box_area,
    calculate_circle_area,
    check_circle_overlap,
    point_in_rect
)

# 2. Importa la funzione per i colori dei sensori
from .sensor_utils import check_sensor_color

# 3. Importa la funzione per la rimozione sfondo
from .background_removal import remove_background_from_pil

# (Opzionale) Se hai un file file_utils.py con le funzioni di upload, scommenta sotto:
# from .file_utils import allowed_file, save_uploaded_file, encode_image_to_base64