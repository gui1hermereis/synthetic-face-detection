import pytest
import torch
import torch.nn as nn

from src.errors.exceptions import ModelInferenceError
from src.services.model_service import ModelService


NORMALIZATION = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}


def artifact(architecture="resnet34", **overrides):
    value = {
        "architecture": architecture,
        "image_size": 224,
        "input_channels": 3,
        "class_names": ["fake", "real"],
        "class_to_index": {"fake": 0, "real": 1},
        "normalization": NORMALIZATION,
        "fake_probability_threshold": 0.2,
        "threshold_selection": {"dataset": "validation", "metric": "fbeta", "beta": 2, "positive_class": "fake"},
        "face_crop": {"detector": "YuNet face_detection_yunet_2023mar.onnx", "score_threshold": 0.9, "single_face_only": True, "margin": 0.15, "crop_shape": "square"},
    }
    value.update(overrides)
    return value


def service(tmp_path, checkpoint="configured.pt", fallback="resnet34"):
    return ModelService({"MODEL_WEIGHTS_PATH": str(tmp_path / checkpoint), "MODEL_ARCHITECTURE": fallback,
                         "MODEL_INPUT_SIZE": 224, "MODEL_CLASS_NAMES": ["fake", "real"]})


@pytest.mark.parametrize("architecture", ["resnet18", "resnet34", "resnet50"])
def test_accepts_supported_rgb_architectures(architecture, tmp_path):
    model = service(tmp_path)
    model._artifact = artifact(architecture)
    metadata = model._metadata_from_artifact()
    assert metadata["architecture"] == architecture
    assert metadata["image_size"] == 224
    assert len(metadata["normalization"]["mean"]) == 3


@pytest.mark.parametrize("field,value", [("architecture", "resnet101"), ("image_size", 256), ("input_channels", 4)])
def test_rejects_unsupported_architecture_dimensions_and_fft(field, value, tmp_path):
    model = service(tmp_path)
    model._artifact = artifact(**{field: value})
    with pytest.raises(ModelInferenceError):
        model._metadata_from_artifact()


def test_model_weights_path_selects_checkpoint_and_metadata_architecture_wins(tmp_path):
    chosen = tmp_path / "resnet18.pt"
    torch.save({"state_dict": {}, **artifact("resnet18")}, chosen)
    model = ModelService({"MODEL_WEIGHTS_PATH": str(chosen), "MODEL_ARCHITECTURE": "resnet50",
                          "MODEL_INPUT_SIZE": 224, "MODEL_CLASS_NAMES": ["fake", "real"]})
    assert model.metadata["architecture"] == "resnet18"
    assert model._weights_path == chosen


def test_legacy_checkpoint_uses_config_architecture_fallback(tmp_path):
    path = tmp_path / "legacy.pt"
    torch.save({"state_dict": {}}, path)
    model = service(tmp_path, "legacy.pt", fallback="resnet50")
    assert model.metadata["architecture"] == "resnet50"


@pytest.mark.parametrize("field,value", [
    ("class_names", ["real", "fake"]),
    ("class_to_index", {"fake": 1, "real": 0}),
    ("normalization", {"mean": [0, 0], "std": [1, 1]}),
])
def test_class_order_mapping_and_rgb_normalization_are_checkpoint_contract(field, value, tmp_path):
    model = service(tmp_path)
    updates = {field: value}
    if field == "class_names":
        updates["class_to_index"] = {"real": 0, "fake": 1}
    model._artifact = artifact(**updates)
    if field == "class_names":
        assert model._metadata_from_artifact()[field] == value
    else:
        with pytest.raises(ModelInferenceError):
            model._metadata_from_artifact()


@pytest.mark.parametrize("shape", [(1, 1, 224, 224), (1, 3, 256, 256), (2, 3, 224, 224)])
def test_rejects_bad_tensor_shape(shape, tmp_path):
    model = service(tmp_path)
    model._artifact = artifact()
    with pytest.raises(ModelInferenceError):
        model._validate_tensor(torch.zeros(shape))


@pytest.mark.parametrize("architecture", ["resnet18", "resnet34", "resnet50"])
def test_loads_state_dict_for_architecture(architecture, tmp_path):
    from torchvision import models
    factory = getattr(models, architecture)
    expected = factory(weights=None)
    expected.fc = nn.Linear(expected.fc.in_features, 2)
    model = service(tmp_path)
    model._artifact = artifact(architecture, state_dict=expected.state_dict())
    model._metadata = model._metadata_from_artifact()
    loaded = model._build_model()
    assert loaded.fc.out_features == 2
    assert loaded.training is False


def test_rejects_state_dict_incompatible_with_architecture(tmp_path):
    model = service(tmp_path)
    model._artifact = artifact("resnet18", state_dict={"not_a_weight": torch.tensor(1)})
    model._metadata = model._metadata_from_artifact()
    with pytest.raises(ModelInferenceError, match="carregar os pesos"):
        model._build_model()


@pytest.mark.parametrize("threshold,score,expected", [(0.37, 0.37, "fake"), (0.37, 0.36, "real"), (0.2, 0.3, "fake")])
def test_prediction_uses_fake_threshold_not_argmax(threshold, score, expected, tmp_path):
    model = service(tmp_path)
    model._artifact = artifact(fake_probability_threshold=threshold)
    model._metadata = model._metadata_from_artifact()
    class FixedLogits(nn.Module):
        def forward(self, _tensor):
            return torch.log(torch.tensor([[score, 1-score]]))
    model._model = FixedLogits()
    result = model.predict(torch.zeros((1, 3, 224, 224)))
    assert result["label"] == expected
    assert result["probability_fake"] == pytest.approx(score, abs=1e-6)
    assert result["decision_threshold"] == threshold
    assert result["decision"]["threshold"] == threshold


def test_metadata_threshold_selection_and_logits_argmax_disagreement(tmp_path):
    model = service(tmp_path)
    model._artifact = artifact(fake_probability_threshold=0.2)
    model._metadata = model._metadata_from_artifact()
    class FixedLogits(nn.Module):
        def forward(self, _tensor):
            return torch.log(torch.tensor([[0.3, 0.7]]))
    model._model = FixedLogits()
    result = model.predict(torch.zeros((1, 3, 224, 224)))
    assert result["label"] == "fake"
    assert result["probability_fake"] == pytest.approx(0.3, abs=1e-6)
    assert result["probability_real"] == pytest.approx(0.7, abs=1e-6)
    assert result["confidence"] == pytest.approx(0.3, abs=1e-6)
    assert result["probabilities"] == pytest.approx({"fake": 0.3, "real": 0.7}, abs=1e-6)
