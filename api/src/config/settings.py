import os
from pathlib import Path

def load_env_file() -> None:
    env_path = Path(__file__).resolve().parents[2] / ".env"
    if not env_path.exists():
        return

    for line in env_path.read_text().splitlines():
        raw_line = line.strip()
        if not raw_line or raw_line.startswith("#") or "=" not in raw_line:
            continue
        key, value = raw_line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())

load_env_file()

class Settings:
    BASE_DIR = Path(__file__).resolve().parents[2]

    SECRET_KEY = os.getenv("SECRET_KEY", "change-me-in-production")
    API_KEY = os.getenv("API_KEY", "")
    MODEL_WEIGHTS_PATH = str((BASE_DIR / "models" / "resnet50_rgb_256.pth").resolve())
    MODEL_METADATA_PATH = str((BASE_DIR / "models" / "model_metadata.json").resolve())
    MODEL_INPUT_SIZE = int(os.getenv("MODEL_INPUT_SIZE", "256"))
    MODEL_CLASS_NAMES = [item.strip() for item in os.getenv("MODEL_CLASS_NAMES", "real,synthetic").split(",") if item.strip()]
    MAX_CONTENT_LENGTH = int(float(os.getenv("MAX_CONTENT_LENGTH_MB", "10")) * 1024 * 1024)
    MIN_IMAGE_WIDTH = int(os.getenv("MIN_IMAGE_WIDTH", "256"))
    MIN_IMAGE_HEIGHT = int(os.getenv("MIN_IMAGE_HEIGHT", "256"))
    MIN_FACE_SIZE = int(os.getenv("MIN_FACE_SIZE", "80"))
    BLUR_THRESHOLD = float(os.getenv("BLUR_THRESHOLD", "90.0"))
    CONTRAST_THRESHOLD = float(os.getenv("CONTRAST_THRESHOLD", "25.0"))