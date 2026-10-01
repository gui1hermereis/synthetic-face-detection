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
