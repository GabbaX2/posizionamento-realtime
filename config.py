import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-key-123-change-in-production'
    UPLOAD_FOLDER = 'uploads'
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

    # Mediapipe configuration
    MODEL_COMPLEXITY = 2
    MIN_DETECTION_CONFIDENCE = 0.5
    MIN_TRACKING_CONFIDENCE = 0.5

    # Sensor Detection
    MIN_SENSOR_AREA = 100
    MAX_SENSOR_AREA = 5000
    COLOR_TOLERANCE = 20

    # Validation
    POSITION_TOLERANCE = 25  # pixels
    ANGLE_TOLERANCE = 15  # degrees
