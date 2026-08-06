import io

from src.services.face_detector import FaceDetectionResult

class DummyModelService:
    def __init__(self, _config):
        pass

    def predict(self, _tensor):
        return {
            "label": "synthetic",
            "confidence": 0.987654,
            "probabilities": {"real": 0.012346, "synthetic": 0.987654},
            "model": {
                "name": "resnet50",
                "input_size": 256,
                "device": "cpu",
                "weights_path": "/tmp/fake-model.pth",
            },
            "training_metrics": {"accuracy": 0.95},
        }

def test_health_check(client):
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["status"] == "ok"

def test_analyze_requires_api_key(client):
    response = client.post("/api/v1/images/analyze")
    assert response.status_code == 401
    assert response.get_json()["error"]["code"] == "authentication_error"

def test_analyze_success(client, sample_image_bytes, monkeypatch):
    monkeypatch.setattr("src.routes.analysis.ModelService", DummyModelService)
    monkeypatch.setattr(
        "src.services.analysis_service.FaceDetector.ensure_single_face",
        lambda self, image_bgr: FaceDetectionResult(faces_detected=1, bounding_boxes=[[10, 20, 100, 100]]),
    )
    monkeypatch.setattr(
        "src.services.analysis_service.ImageQualityInspector.inspect",
        lambda self, image_bgr: {
            "blur_score": 250.0,
            "contrast_score": 55.0,
            "blur_threshold": 90.0,
            "contrast_threshold": 25.0,
        },
    )

    response = client.post(
        "/api/v1/images/analyze",
        headers={"X-API-Key": "test-key"},
        data={"image": (io.BytesIO(sample_image_bytes), "face.jpg")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["prediction"]["label"] == "synthetic"
    assert payload["face_detection"]["faces_detected"] == 1

def test_analyze_without_image_returns_400(client):
    response = client.post(
        "/api/v1/images/analyze",
        headers={"X-API-Key": "test-key"},
        data={},
        content_type="multipart/form-data",
    )
    assert response.status_code == 400
    assert response.get_json()["error"]["code"] == "invalid_payload"