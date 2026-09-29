import os
from pathlib import Path

from dotenv import load_dotenv

API_DIR = Path(__file__).resolve().parents[2]
load_dotenv(API_DIR / ".env", override=False)

def path_from_env(variable_name: str, default_path: Path) -> str:
    """Resolve caminhos do .env em relação à pasta api/."""
    configured_path = Path(os.getenv(variable_name, str(default_path)))
    if not configured_path.is_absolute():
        configured_path = API_DIR / configured_path
    return str(configured_path.resolve())

class Settings:
    """Configurações centralizadas da API de inferência."""

    BASE_DIR = API_DIR
    PROJECT_DIR = BASE_DIR.parent
    MODELS_DIR = PROJECT_DIR / "models"

    SECRET_KEY = os.getenv("SECRET_KEY", "")
    API_KEY = os.getenv("API_KEY", "")
    MAX_CONTENT_LENGTH = int(float(os.getenv("MAX_CONTENT_LENGTH_MB", "10")) * 1024 * 1024)

    MIN_IMAGE_WIDTH = int(os.getenv("MIN_IMAGE_WIDTH", "256"))
    MIN_IMAGE_HEIGHT = int(os.getenv("MIN_IMAGE_HEIGHT", "256"))
    MIN_FACE_SIZE = int(os.getenv("MIN_FACE_SIZE", "80"))

    FACE_DETECTOR_MODEL_PATH = path_from_env(
        "FACE_DETECTOR_MODEL_PATH",
        MODELS_DIR / "face_detection_yunet_2023mar.onnx",
    )

    BLUR_THRESHOLD = float(os.getenv("BLUR_THRESHOLD", "90.0"))
    CONTRAST_THRESHOLD = float(os.getenv("CONTRAST_THRESHOLD", "25.0"))

    MODEL_WEIGHTS_PATH = path_from_env(
        "MODEL_WEIGHTS_PATH",
        MODELS_DIR / "resnet34_224_source_holdout_seed42.pt",
    )

    # Fallback para checkpoints antigos.
    MODEL_ARCHITECTURE = os.getenv("MODEL_ARCHITECTURE", "resnet34")
    MODEL_INPUT_SIZE = int(os.getenv("MODEL_INPUT_SIZE", "224"))
    MODEL_CLASS_NAMES = [
        name.strip()
        for name in os.getenv("MODEL_CLASS_NAMES", "fake,real").split(",")
        if name.strip()
    ]