import io
import sys
from pathlib import Path

import pytest
from PIL import Image

API_DIR = Path(__file__).resolve().parents[1]
if str(API_DIR) not in sys.path:
    sys.path.insert(0, str(API_DIR))

from server import create_app


class FakeFaceDetector:
    """Evita carregar o modelo ONNX nos testes de rota."""

    def __init__(self, _config):
        pass

    def ensure_single_face(self, _image_bgr):
        from src.services.face_detector import FaceDetectionResult

        return FaceDetectionResult(
            faces_detected=1,
            bounding_boxes=[[10, 20, 100, 100]],
        )

@pytest.fixture
def app(monkeypatch, tmp_path):
    monkeypatch.setattr("server.FaceDetector", FakeFaceDetector)
    weights_path = tmp_path / "resnet50_rgb_256.pth"
    weights_path.touch()

    app = create_app(
        {
            "TESTING": True,
            "API_KEY": "test-key",
            "MODEL_WEIGHTS_PATH": str(weights_path),
        }
    )
    return app

@pytest.fixture
def client(app):
    return app.test_client()

@pytest.fixture
def make_image_bytes():
    def build(size=(512, 512), color=(180, 120, 90)):
        image = Image.new("RGB", size, color=color)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")
        return buffer.getvalue()

    return build


@pytest.fixture
def sample_image_bytes(make_image_bytes):
    return make_image_bytes()
