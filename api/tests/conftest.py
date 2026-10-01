import io
import sys
from pathlib import Path

import pytest
from PIL import Image

API_DIR = Path(__file__).resolve().parents[1]
if str(API_DIR) not in sys.path:
    sys.path.insert(0, str(API_DIR))

from server import create_app
from src.errors.exceptions import ModelInferenceError
from src.services.face_detector import FaceDetectionResult


OFFICIAL_METADATA = {
    "architecture": "resnet34",
    "image_size": 224,
    "class_names": ["fake", "real"],
    "class_to_index": {"fake": 0, "real": 1},
    "normalization": {
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    },
    "fake_probability_threshold": 0.2,
    "threshold_selection": {
        "dataset": "validation",
        "metric": "fbeta",
        "beta": 2,
        "positive_class": "fake",
    },
    "face_crop": {
        "detector": "YuNet face_detection_yunet_2023mar.onnx",
        "score_threshold": 0.9,
        "single_face_only": True,
        "margin": 0.15,
        "crop_shape": "square",
    },
}


class FakeModelService:
    """Modelo em memória compatível com o artefato oficial, sem GPU/pesos reais."""

    def __init__(self, _config):
        self.metadata = OFFICIAL_METADATA.copy()
        self.fake_probability = 0.8
        self.error = None
        self.last_tensor_shape = None

    def predict(self, tensor):
        self.last_tensor_shape = list(tensor.shape)
        if self.error is not None:
            raise self.error

        fake_probability = self.fake_probability
        real_probability = 1 - fake_probability
        threshold = self.metadata["fake_probability_threshold"]
        label = "fake" if fake_probability >= threshold else "real"
        confidence = fake_probability if label == "fake" else real_probability
        return {
            "label": label,
            "confidence": round(confidence, 6),
            "probability_fake": round(fake_probability, 6),
            "probability_real": round(real_probability, 6),
            "decision_threshold": threshold,
            "probabilities": {
                "fake": round(fake_probability, 6),
                "real": round(real_probability, 6),
            },
            "decision": {
                "positive_class": "fake",
                "rule": "probability_fake >= threshold",
                "threshold": threshold,
                "threshold_selection": self.metadata["threshold_selection"],
            },
            "model": {
                "name": "resnet34",
                "input_size": 224,
                "device": "cpu",
                "weights": "resnet34_224.pt",
            },
        }


class FakeFaceDetector:
    def __init__(self, _config, _metadata):
        self.result = FaceDetectionResult(1, [[10, 20, 100, 100]])
        self.error = None

    def ensure_single_face(self, _image_bgr):
        if self.error is not None:
            raise self.error
        return self.result


@pytest.fixture
def app(monkeypatch):
    monkeypatch.setattr("server.ModelService", FakeModelService)
    monkeypatch.setattr("server.FaceDetector", FakeFaceDetector)
    monkeypatch.setattr(
        "src.services.analysis_service.ImageQualityInspector.inspect",
        lambda _self, _image: {
            "blur_score": 250.0,
            "contrast_score": 55.0,
            "blur_threshold": 90.0,
            "contrast_threshold": 25.0,
        },
    )
    return create_app({"TESTING": True, "API_KEY": "test-key"})


@pytest.fixture
def client(app):
    return app.test_client()


@pytest.fixture
def make_image_bytes():
    def build(size=(512, 512), color=(180, 120, 90), image_format="JPEG"):
        image = Image.new("RGB", size, color=color)
        buffer = io.BytesIO()
        image.save(buffer, format=image_format)
        return buffer.getvalue()

    return build


@pytest.fixture
def sample_image_bytes(make_image_bytes):
    return make_image_bytes()


@pytest.fixture
def model_service(app):
    return app.extensions["model_service"]


@pytest.fixture
def model_inference_error():
    return ModelInferenceError(
        "Não foi possível concluir a análise da imagem.",
        details={"weights_path": "/internal/models/resnet34_224.pt", "trace": "secret"},
    )
