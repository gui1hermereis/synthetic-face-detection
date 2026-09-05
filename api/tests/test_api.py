import io

from src.errors.exceptions import FaceValidationError, ImageQualityError


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
                "weights": "fake-model.pth",
            },
        }


def post_image(client, image_bytes, filename="face.jpg", api_key="test-key"):
    return client.post(
        "/api/v1/images/analyze",
        headers={"X-API-Key": api_key},
        data={"image": (io.BytesIO(image_bytes), filename)},
        content_type="multipart/form-data",
    )


def test_health_check(client):
    response = client.get("/api/v1/health")

    assert response.status_code == 200
    assert response.get_json()["status"] == "ok"
    assert response.get_json()["model_available"] is True
    assert response.get_json()["model_weights"] == "resnet50_rgb_256.pth"


def test_health_returns_503_when_weights_are_missing(client, app, tmp_path):
    app.config["MODEL_WEIGHTS_PATH"] = str(tmp_path / "missing-model.pth")

    response = client.get("/api/v1/health")

    assert response.status_code == 503
    assert response.get_json()["status"] == "error"
    assert response.get_json()["model_available"] is False


def test_analyze_requires_api_key(client):
    response = client.post("/api/v1/images/analyze")

    assert response.status_code == 401
    assert response.get_json()["error"]["code"] == "authentication_error"


def test_analyze_rejects_invalid_api_key(client, sample_image_bytes):
    response = post_image(client, sample_image_bytes, api_key="wrong-key")

    assert response.status_code == 401
    assert response.get_json()["error"]["code"] == "authentication_error"


def test_analyze_without_image_returns_400(client):
    response = client.post(
        "/api/v1/images/analyze",
        headers={"X-API-Key": "test-key"},
        data={},
        content_type="multipart/form-data",
    )

    assert response.status_code == 400
    assert response.get_json()["error"]["code"] == "invalid_payload"


def test_analyze_rejects_invalid_image(client):
    response = post_image(client, b"not-an-image")

    assert response.status_code == 422
    assert response.get_json()["error"]["code"] == "image_validation_error"


def test_analyze_rejects_insufficient_resolution(client, make_image_bytes):
    response = post_image(client, make_image_bytes(size=(255, 255)))

    assert response.status_code == 422
    payload = response.get_json()
    assert payload["error"]["code"] == "image_validation_error"
    assert payload["error"]["details"]["minimum_width"] == 256


def test_analyze_rejects_image_without_faces(client, app, sample_image_bytes, monkeypatch):
    def no_face(_image_bgr):
        raise FaceValidationError("Nenhum rosto detectado.", {"faces_detected": 0})

    monkeypatch.setattr(app.extensions["face_detector"], "ensure_single_face", no_face)

    response = post_image(client, sample_image_bytes)

    assert response.status_code == 422
    assert response.get_json()["error"]["code"] == "face_validation_error"
    assert response.get_json()["error"]["details"]["faces_detected"] == 0


def test_analyze_rejects_multiple_faces(client, app, sample_image_bytes, monkeypatch):
    def multiple_faces(_image_bgr):
        raise FaceValidationError(
            "Mais de um rosto detectado.",
            {"faces_detected": 2, "bounding_boxes": [[1, 2, 30, 30], [40, 50, 30, 30]]},
        )

    monkeypatch.setattr(app.extensions["face_detector"], "ensure_single_face", multiple_faces)

    response = post_image(client, sample_image_bytes)

    assert response.status_code == 422
    payload = response.get_json()
    assert payload["error"]["code"] == "face_validation_error"
    assert payload["error"]["details"]["faces_detected"] == 2


def test_analyze_rejects_blurry_face(client, sample_image_bytes, monkeypatch):
    def blurry_face(_image_bgr):
        raise ImageQualityError("Face borrada.", {"blur_score": 20.0})

    monkeypatch.setattr(
        "src.services.analysis_service.ImageQualityInspector.inspect",
        lambda _self, image_bgr: blurry_face(image_bgr),
    )

    response = post_image(client, sample_image_bytes)

    assert response.status_code == 422
    payload = response.get_json()
    assert payload["error"]["code"] == "image_quality_error"
    assert payload["error"]["details"]["blur_score"] == 20.0


def test_analyze_success_preprocesses_the_face_crop(client, sample_image_bytes, monkeypatch):
    monkeypatch.setattr("src.routes.analysis.ModelService", DummyModelService)
    captured = {}

    def inspect_crop(_self, image_bgr):
        captured["shape"] = image_bgr.shape
        return {
            "blur_score": 250.0,
            "contrast_score": 55.0,
            "blur_threshold": 90.0,
            "contrast_threshold": 25.0,
        }

    monkeypatch.setattr("src.services.analysis_service.ImageQualityInspector.inspect", inspect_crop)

    response = post_image(client, sample_image_bytes)

    assert response.status_code == 200
    payload = response.get_json()
    assert captured["shape"] == (130, 130, 3)
    assert payload["prediction"]["label"] == "synthetic"
    assert payload["face_detection"] == {
        "faces_detected": 1,
        "bounding_boxes": [[10, 20, 100, 100]],
    }
