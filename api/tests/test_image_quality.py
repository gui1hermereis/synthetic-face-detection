import numpy as np
import pytest

from src.errors.exceptions import ImageQualityError
from src.services.image_quality import ImageQualityInspector


def test_quality_inspector_rejects_blurry_face():
    inspector = ImageQualityInspector({"BLUR_THRESHOLD": 90.0, "CONTRAST_THRESHOLD": 25.0})
    image = np.full((224, 224, 3), 128, dtype=np.uint8)
    with pytest.raises(ImageQualityError, match="desfocada"):
        inspector.inspect(image)


def test_quality_inspector_rejects_low_contrast_face():
    inspector = ImageQualityInspector({"BLUR_THRESHOLD": 1.0, "CONTRAST_THRESHOLD": 25.0})
    grid = np.indices((224, 224)).sum(axis=0) % 2
    image = np.where(grid[..., None] == 0, 120, 121).astype(np.uint8)
    image = np.repeat(image, 3, axis=2)
    with pytest.raises(ImageQualityError, match="contraste muito baixo"):
        inspector.inspect(image)


def test_quality_inspector_resizes_with_area_and_reports_configured_thresholds(monkeypatch):
    import src.services.image_quality as module

    calls = []
    original_resize = module.cv2.resize

    def capture_resize(image, size, interpolation):
        calls.append((size, interpolation))
        return original_resize(image, size, interpolation=interpolation)

    monkeypatch.setattr(module.cv2, "resize", capture_resize)
    image = np.random.default_rng(7).integers(0, 256, (320, 280, 3), dtype=np.uint8)
    result = ImageQualityInspector({"BLUR_THRESHOLD": 0.0, "CONTRAST_THRESHOLD": 0.0}).inspect(image)

    assert calls == [((224, 224), module.cv2.INTER_AREA)]
    assert result["blur_threshold"] == 0.0
    assert result["contrast_threshold"] == 0.0
    assert result["blur_score"] >= 0
    assert result["contrast_score"] >= 0
