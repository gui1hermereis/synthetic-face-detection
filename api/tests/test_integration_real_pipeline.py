"""Opt-in integration tests using real YuNet and real model weights.

Set SF_API_INTEGRATION_IMAGE to a permitted, supported photo containing exactly
one face. Tests are skipped with an explicit reason when any artifact is absent.
"""
import os
from pathlib import Path

import pytest

from server import create_app


API_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = API_DIR.parent
WEIGHTS = Path(os.getenv("MODEL_WEIGHTS_PATH", PROJECT_DIR / "models/resnet34_224.pt"))
if not WEIGHTS.is_absolute():
    WEIGHTS = (API_DIR / WEIGHTS).resolve()
YUNET = Path(os.getenv("FACE_DETECTOR_MODEL_PATH", PROJECT_DIR / "models/face_detection_yunet_2023mar.onnx"))
if not YUNET.is_absolute():
    YUNET = (API_DIR / YUNET).resolve()
IMAGE = Path(os.getenv("SF_API_INTEGRATION_IMAGE", ""))

reason = "requires real model checkpoint, YuNet ONNX, and SF_API_INTEGRATION_IMAGE containing exactly one face"
pytestmark = [pytest.mark.integration, pytest.mark.skipif(not (WEIGHTS.is_file() and YUNET.is_file() and IMAGE.is_file()), reason=reason)]


def test_real_yunet_and_checkpoint_complete_pipeline():
    app = create_app({"TESTING": True, "API_KEY": "integration-key",
                      "MODEL_WEIGHTS_PATH": str(WEIGHTS), "FACE_DETECTOR_MODEL_PATH": str(YUNET)})
    with app.test_client() as client, IMAGE.open("rb") as image:
        response = client.post("/api/v1/images/analyze", headers={"X-API-Key": "integration-key"},
                               data={"image": (image, IMAGE.name)}, content_type="multipart/form-data")
    assert response.status_code == 200, response.get_json()
    body = response.get_json()
    assert body["face_detection"]["faces_detected"] == 1
    assert body["prediction"]["model"]["name"] == app.extensions["model_service"].metadata["architecture"]
    assert body["prediction"]["label"] in {"fake", "real"}
    assert 0 <= body["prediction"]["probability_fake"] <= 1
