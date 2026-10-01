import io

import pytest

from src.errors.exceptions import FaceValidationError, ImageQualityError
from src.services.face_detector import FaceDetectionResult


def post_image(client, image_bytes, filename="face.jpg", api_key="test-key"):
    return client.post(
        "/api/v1/images/analyze",
        headers={"X-API-Key": api_key},
        data={"image": (io.BytesIO(image_bytes), filename)},
        content_type="multipart/form-data",
    )


def test_health_reports_official_model_contract(client):
    response = client.get("/api/v1/health")

    assert response.status_code == 200
    assert response.get_json() == {
        "status": "ok",
        "message": "API Flask inicializada com sucesso.",
        "model_available": True,
        "model": {
            "architecture": "resnet34",
            "input_size": 224,
            "input_channels": 3,
            "class_names": ["fake", "real"],
        },
    }


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


@pytest.mark.parametrize(
    ("image_format", "filename", "expected_format"),
    [("JPEG", "face.jpg", "JPEG"), ("JPEG", "face.jpeg", "JPEG"), ("PNG", "face.png", "PNG")],
)
def test_analyze_accepts_supported_formats(client, make_image_bytes, image_format, filename, expected_format):
    response = post_image(client, make_image_bytes(image_format=image_format), filename)
    assert response.status_code == 200
    assert response.get_json()["image"]["format"] == expected_format


@pytest.mark.parametrize(
    ("payload", "filename", "expected_format"),
    [
        (b"%PDF-1.7 fake", "face.pdf", None),
        (b"not a text image", "face.txt", None),
        ("WEBP", "face.webp", "WEBP"),
    ],
)
def test_analyze_rejects_unsupported_formats(client, make_image_bytes, payload, filename, expected_format):
    image_bytes = make_image_bytes(image_format=payload) if payload == "WEBP" else payload
    response = post_image(client, image_bytes, filename)
    assert response.status_code == 422
    body = response.get_json()["error"]
    assert body["code"] == "image_validation_error"
    assert "JPG, JPEG ou PNG" in body["message"] or "não é uma imagem válida" in body["message"]
    if expected_format:
        assert body["details"]["received_format"] == expected_format


@pytest.mark.parametrize(
    ("image_bytes", "filename"),
    [(b"not-an-image", "face.jpg"), (bytes([0xFF, 0xD8, 0xFF, 0xE0]) + b"truncated", "face.png")],
)
def test_analyze_rejects_corrupted_or_renamed_image(client, image_bytes, filename):
    response = post_image(client, image_bytes, filename)
    assert response.status_code == 422
    assert response.get_json()["error"]["code"] == "image_validation_error"


def test_analyze_rejects_insufficient_resolution(client, make_image_bytes):
    response = post_image(client, make_image_bytes(size=(255, 255)))
    assert response.status_code == 422
    assert response.get_json()["error"]["details"]["minimum_width"] == 256


def test_analyze_rejects_image_without_faces(client, app, sample_image_bytes):
    app.extensions["face_detector"].error = FaceValidationError(
        "Não foi detectado um rosto humano na imagem enviada.", {"faces_detected": 0}
    )
    response = post_image(client, sample_image_bytes)
    assert response.status_code == 422
    assert response.get_json()["error"]["details"]["faces_detected"] == 0


def test_analyze_rejects_multiple_faces(client, app, sample_image_bytes):
    app.extensions["face_detector"].error = FaceValidationError(
        "A imagem enviada contém mais de um rosto humano.",
        {"faces_detected": 2, "bounding_boxes": [[1, 2, 30, 30], [40, 50, 30, 30]]},
    )
    response = post_image(client, sample_image_bytes)
    assert response.status_code == 422
    assert response.get_json()["error"]["details"]["faces_detected"] == 2


@pytest.mark.parametrize("message, details", [
    ("A imagem enviada está muito tremida ou desfocada para uma análise confiável.", {"blur_score": 20.0}),
    ("A imagem enviada possui contraste muito baixo para uma análise confiável.", {"contrast_score": 3.0}),
])
def test_analyze_rejects_low_quality_face(client, app, sample_image_bytes, monkeypatch, message, details):
    def reject(_self, _image):
        raise ImageQualityError(message, details)

    monkeypatch.setattr("src.services.analysis_service.ImageQualityInspector.inspect", reject)
    response = post_image(client, sample_image_bytes)
    assert response.status_code == 422
    assert response.get_json()["error"] == {
        "code": "image_quality_error", "message": message, "details": details
    }


def test_analyze_returns_fake_result_and_rgb_224_tensor(client, sample_image_bytes, model_service):
    model_service.fake_probability = 0.8
    response = post_image(client, sample_image_bytes)
    assert response.status_code == 200
    prediction = response.get_json()["prediction"]
    assert prediction["label"] == "fake"
    assert prediction["probability_fake"] == 0.8
    assert prediction["probability_real"] == 0.2
    assert prediction["decision_threshold"] == 0.2
    assert model_service.last_tensor_shape == [1, 3, 224, 224]


def test_analyze_returns_real_result_below_threshold(client, sample_image_bytes, model_service):
    model_service.fake_probability = 0.19
    response = post_image(client, sample_image_bytes)
    prediction = response.get_json()["prediction"]
    assert prediction["label"] == "real"
    assert prediction["probabilities"] == {"fake": 0.19, "real": 0.81}


def test_analyze_uses_fake_threshold_on_boundary(client, sample_image_bytes, model_service):
    model_service.fake_probability = 0.2
    response = post_image(client, sample_image_bytes)
    assert response.get_json()["prediction"]["label"] == "fake"


def test_result_contains_frontend_contract(client, sample_image_bytes):
    response = post_image(client, sample_image_bytes)
    body = response.get_json()
    assert {"filename", "image", "face_detection", "quality", "prediction"} <= body.keys()
    assert {
        "label", "confidence", "probability_fake", "probability_real",
        "decision_threshold", "probabilities", "decision", "model",
    } <= body["prediction"].keys()


def test_images_are_processed_in_memory_without_persistence(client, sample_image_bytes, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    before = list(tmp_path.iterdir())
    response = post_image(client, sample_image_bytes)
    assert response.status_code == 200
    assert list(tmp_path.iterdir()) == before


def test_model_error_does_not_expose_internal_details(client, sample_image_bytes, model_service, model_inference_error):
    model_service.error = model_inference_error
    response = post_image(client, sample_image_bytes)
    assert response.status_code == 500
    error = response.get_json()["error"]
    assert error["code"] == "model_inference_error"
    assert error["details"] == {}
    assert "/internal/" not in response.get_data(as_text=True)
