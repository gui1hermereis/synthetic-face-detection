import os
from pathlib import Path


class Settings:
    """Configuracoes centralizadas da API de inferencia."""

    BASE_DIR = Path(__file__).resolve().parents[2]
    MODELS_DIR = BASE_DIR / "models"

    SECRET_KEY = os.getenv("SECRET_KEY", "")
    API_KEY = os.getenv("API_KEY", "")
    MAX_CONTENT_LENGTH = int(float(os.getenv("MAX_CONTENT_LENGTH_MB", "10")) * 1024 * 1024)

    MIN_IMAGE_WIDTH = int(os.getenv("MIN_IMAGE_WIDTH", "256"))
    MIN_IMAGE_HEIGHT = int(os.getenv("MIN_IMAGE_HEIGHT", "256"))

    FACE_DETECTOR_MODEL_PATH = os.getenv(
        "FACE_DETECTOR_MODEL_PATH",
        str(MODELS_DIR / "face_detection_yunet_2023mar.onnx"),
    )
    FACE_DETECTOR_SCORE_THRESHOLD = float(os.getenv("FACE_DETECTOR_SCORE_THRESHOLD", "0.9"))
    MIN_FACE_SIZE = int(os.getenv("MIN_FACE_SIZE", "80"))
    FACE_CROP_PADDING_RATIO = float(os.getenv("FACE_CROP_PADDING_RATIO", "0.15"))

    BLUR_THRESHOLD = float(os.getenv("BLUR_THRESHOLD", "90.0"))
    CONTRAST_THRESHOLD = float(os.getenv("CONTRAST_THRESHOLD", "25.0"))

    MODEL_WEIGHTS_PATH = os.getenv(
        "MODEL_WEIGHTS_PATH",
        str(MODELS_DIR / "resnet50_rgb_256.pth"),
    )
    MODEL_INPUT_SIZE = int(os.getenv("MODEL_INPUT_SIZE", "256"))
    MODEL_CLASS_NAMES = ["real", "synthetic"]
