import io
import sys
from pathlib import Path

import pytest
from PIL import Image

API_DIR = Path(__file__).resolve().parents[1]
if str(API_DIR) not in sys.path:
    sys.path.insert(0, str(API_DIR))

from src import create_app

@pytest.fixture
def app():
    app = create_app(
        {
            "TESTING": True,
            "API_KEY": "test-key",
            "MODEL_WEIGHTS_PATH": str(API_DIR / "models" / "resnet50_rgb_256.pth"),
            "MODEL_METADATA_PATH": str(API_DIR / "models" / "model_metadata.json"),
        }
    )
    return app

@pytest.fixture
def client(app):
    return app.test_client()

@pytest.fixture
def sample_image_bytes():
    image = Image.new("RGB", (512, 512), color=(180, 120, 90))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    buffer.seek(0)
    return buffer.read()