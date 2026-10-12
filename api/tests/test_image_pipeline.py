import numpy as np
import pytest
import torch
from PIL import Image

from src.errors.exceptions import FaceValidationError, ImageValidationError
from src.services.face_detector import FaceDetectionResult
from src.services.image_loader import ImageLoader
from src.services.preprocessor import ModelPreprocessor


def test_loader_rejects_empty_and_over_limit():
    loader = ImageLoader({"MIN_IMAGE_WIDTH": 1, "MIN_IMAGE_HEIGHT": 1, "MAX_CONTENT_LENGTH": 3})
    with pytest.raises(ImageValidationError):
        loader.load(b"", "x.jpg")
    with pytest.raises(ImageValidationError):
        loader.load(b"1234", "x.jpg")


def test_loader_exif_orientation_applied(make_image_bytes):
    from io import BytesIO
    from PIL import Image as PILImage
    image = PILImage.new("RGB", (300, 500), "red")
    buffer = BytesIO()
    exif = image.getexif()
    exif[274] = 6
    image.save(buffer, format="JPEG", exif=exif)
    loaded = ImageLoader({"MIN_IMAGE_WIDTH": 1, "MIN_IMAGE_HEIGHT": 1, "MAX_CONTENT_LENGTH": 100000}).load(buffer.getvalue(), "rotated.jpg")
    assert (loaded.width, loaded.height) == (500, 300)


def test_preprocessor_resizes_rgb_area_and_normalizes_checkpoint(monkeypatch):
    import src.services.preprocessor as module
    calls = []
    original = module.cv2.resize
    def resize(image, size, interpolation):
        calls.append((size, interpolation, image.copy()))
        return original(image, size, interpolation=interpolation)
    monkeypatch.setattr(module.cv2, "resize", resize)
    image = Image.fromarray(np.full((100, 120, 3), [255, 0, 0], dtype=np.uint8))
    tensor = ModelPreprocessor({"image_size": 224, "normalization": {"mean": [0.5]*3, "std": [0.5]*3}}).transform(image)
    assert tensor.shape == (1, 3, 224, 224)
    assert calls[0][0] == (224, 224)
    assert calls[0][1] == module.cv2.INTER_AREA
    assert torch.allclose(tensor[0, :, 100, 100], torch.tensor([1., -1., -1.]), atol=1e-5)


def test_detector_zero_one_multiple_faces_and_resize_coordinates(fake_yunet):
    detector, yunet = fake_yunet
    image = np.zeros((640, 1280, 3), dtype=np.uint8)
    yunet.faces = None
    with pytest.raises(FaceValidationError) as zero:
        detector.ensure_single_face(image)
    assert zero.value.details["faces_detected"] == 0
    yunet.faces = np.array([[25, 40, 80, 100, *([0] * 10), .99]], dtype=np.float32)
    result = detector.ensure_single_face(image)
    assert yunet.input_size == (320, 160)
    assert result.bounding_boxes == [[100, 160, 320, 400]]
    yunet.faces = np.array([[25, 40, 80, 100, *([0] * 10), .99], [150, 40, 80, 100, *([0] * 10), .98]], dtype=np.float32)
    with pytest.raises(FaceValidationError) as multiple:
        detector.ensure_single_face(image)
    assert multiple.value.details["faces_detected"] == 2


@pytest.mark.parametrize("bbox", [[0, 0, 20, 40], [80, 80, 20, 40]])
def test_square_crop_margin_and_edges_match_training_algorithm(bbox):
    image = np.arange(100 * 100 * 3, dtype=np.uint8).reshape(100, 100, 3)
    result = FaceDetectionResult(1, [bbox])
    crop = result.crop_single_face(image, 0.15)
    x, y, width, height = bbox
    side = min(int(round(max(width, height) * (1 + 2 * .15))), 100, 100)
    left = max(0, min(int(round(x + width / 2 - side / 2)), 100 - side))
    top = max(0, min(int(round(y + height / 2 - side / 2)), 100 - side))
    assert crop.shape == (side, side, 3)
    assert np.array_equal(crop, image[top:top+side, left:left+side])


def test_detector_has_no_min_face_size_filter(fake_yunet):
    detector, yunet = fake_yunet
    yunet.faces = np.array([[10, 10, 3, 4, *([0] * 10), .99]], dtype=np.float32)
    result = detector.ensure_single_face(np.zeros((100, 100, 3), dtype=np.uint8))
    assert result.bounding_boxes == [[10, 10, 3, 4]]
