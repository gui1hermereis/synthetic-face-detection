import os
from pathlib import Path

class Settings:
    BASE_DIR = Path(__file__).resolve().parents[2]

    SECRET_KEY = os.getenv("SECRET_KEY", "")
    API_KEY = os.getenv("API_KEY", "")

    MODEL_WEIGHTS_PATH = str((BASE_DIR / "models" / "resnet50_rgb_256.pth").resolve())
    MODEL_INPUT_SIZE = 256
    MODEL_CLASS_NAMES = ["real", "synthetic"]

    MAX_CONTENT_LENGTH = int(float(os.getenv("MAX_CONTENT_LENGTH_MB", "10")) * 1024 * 1024)

    MIN_IMAGE_WIDTH = int(os.getenv("MIN_IMAGE_WIDTH", "256"))
    MIN_IMAGE_HEIGHT = int(os.getenv("MIN_IMAGE_HEIGHT", "256"))
    MIN_FACE_SIZE = int(os.getenv("MIN_FACE_SIZE", "80"))

    BLUR_THRESHOLD = float(os.getenv("BLUR_THRESHOLD", "90.0"))
    CONTRAST_THRESHOLD = float(os.getenv("CONTRAST_THRESHOLD", "25.0"))
