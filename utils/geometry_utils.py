import numpy as np
from typing import Tuple, List

def calculate_distance(point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
    """
    Calcola distanza Euclidea tra due punti.
    """
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

def calculate_angle(point1: Tuple[float, float], point2: Tuple[float, float], point3: Tuple[float, float]) -> float:
    """
    Calcola l'angolo in gradi tra 3 punti (p2 è il vertice).
    """
    a = np.array(point1)
    b = np.array(point2)
    c = np.array(point3)

    ba = a - b
    bc = c - b

    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)

    # Evita divisione per zero
    if norm_ba == 0 or norm_bc == 0:
        return 0.0

    cosine_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))

    return np.degrees(angle)

def rotate_point(point: Tuple[float, float], center: Tuple[float, float], angle: float) -> Tuple[float, float]:
    """
    Ruota un punto attorno a un centro di un dato angolo (in gradi).
    """
    angle_rad = np.radians(angle)
    x, y = point
    cx, cy = center

    # Traslazione all'origine
    x_translated = x - cx
    y_translated = y - cy

    # Rotazione
    x_rotated = x_translated * np.cos(angle_rad) - y_translated * np.sin(angle_rad)
    y_rotated = x_translated * np.sin(angle_rad) + y_translated * np.cos(angle_rad)

    # Ritraslazione
    x_new = x_rotated + cx
    y_new = y_rotated + cy

    return (x_new, y_new)

def calculate_box_area(w: int, h: int) -> int:
    """Calcola l'area di un rettangolo"""
    return w * h

def calculate_circle_area(radius: float) -> float:
    """Calcola l'area di un cerchio"""
    return np.pi * (radius ** 2)

def check_circle_overlap(c1: Tuple[float, float, float], c2: Tuple[float, float, float]) -> bool:
    """
    Controlla se due cerchi si sovrappongono.
    Input: (x, y, raggio) per entrambi.
    """
    x1, y1, r1 = c1
    x2, y2, r2 = c2
    dist = calculate_distance((x1, y1), (x2, y2))
    return dist < (r1 + r2)

def point_in_rect(point: Tuple[float, float], rect: Tuple[int, int, int, int]) -> bool:
    """
    Controlla se un punto è dentro un rettangolo (x, y, w, h).
    """
    px, py = point
    rx, ry, rw, rh = rect
    return rx <= px <= rx + rw and ry <= py <= ry + rh