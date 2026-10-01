import pytest

from src.errors.exceptions import ModelInferenceError
from src.services.model_service import ModelService


def metadata_service(artifact):
    service = ModelService({
        "MODEL_WEIGHTS_PATH": "unused.pt",
        "MODEL_ARCHITECTURE": "resnet34",
        "MODEL_INPUT_SIZE": 224,
        "MODEL_CLASS_NAMES": ["fake", "real"],
    })
    service._artifact = artifact
    return service


def official_artifact():
    return {
        "architecture": "resnet34",
        "image_size": 224,
        "class_names": ["fake", "real"],
        "class_to_index": {"fake": 0, "real": 1},
        "normalization": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
        "fake_probability_threshold": 0.2,
        "face_crop": {
            "detector": "YuNet face_detection_yunet_2023mar.onnx",
            "score_threshold": 0.9,
            "single_face_only": True,
            "margin": 0.15,
            "crop_shape": "square",
        },
    }


def test_official_model_contract_is_resnet34_rgb_224():
    metadata = metadata_service(official_artifact())._metadata_from_artifact()
    assert metadata["architecture"] == "resnet34"
    assert metadata["image_size"] == 224
    assert len(metadata["normalization"]["mean"]) == 3


@pytest.mark.parametrize("field, value", [
    ("architecture", "resnet50"),
    ("image_size", 256),
    ("input_channels", 4),
])
def test_rejects_non_official_or_fft_artifacts(field, value):
    artifact = official_artifact()
    artifact[field] = value
    with pytest.raises(ModelInferenceError, match="ResNet34 RGB 224x224"):
        metadata_service(artifact)._metadata_from_artifact()
