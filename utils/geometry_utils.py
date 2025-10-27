import numpy as np
from typing import Tuple


def calculate_distance(point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
    """Calcola distanza tra due punti"""
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1] ** 2))


def calculate_angle(point1: Tuple[float, float], point2: Tuple[float, float], point3: Tuple[float, float]) -> float:
    """Calcola l'angolo tra 3 punti"""
    a = np.array(point1)
    b = np.array(point2)
    c = np.array(point3)

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))

    return np.degrees(angle)


def rotate_point(point: Tuple[float, float], center: Tuple[float, float], angle: float) -> Tuple[float, float]:
    """Rotate point around center by given angle (deg)"""
    angle_rad = np.radians(angle)
    x, y = point
    cx, cy = center

    # Translate point to origin
    x_translated = x - cx
    y_translated = y - cy

    # Rotate
    x_rotated = x_translated * np.cos(angle_rad) - y_translated * np.sin(angle_rad)
    y_rotated = x_translated * np.sin(angle_rad) + y_translated * np.cos(angle_rad)

    # Translate back
    x_new = x_rotated + cx
    y_new = y_rotated + cy

    return (x_new, y_new)


def calculate_midpoint(point1: Tuple[float, float], point2: Tuple[float, float]) -> Tuple[float, float]:
    return (point1[0] + point2[0]) / 2, (point1[1] + point2[1]) / 2


def point_in_polygon(point: Tuple[float, float], polygon: list) -> bool:
    """Check if point is inside polygon using ray casting algorithm"""
    x, y = point
    n = len(polygon)
    inside = False

    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y

    return inside
