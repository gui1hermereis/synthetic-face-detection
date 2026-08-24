from werkzeug.datastructures import FileStorage

from ..errors.exceptions import InvalidPayloadError
from .face_detector import FaceDetector
from .image_loader import ImageLoader
from .image_quality import ImageQualityInspector
from .model_service import ModelService
from .preprocessor import ModelPreprocessor

class AnalysisService:
    def __init__(self, config, model_service: ModelService):
        self._image_loader = ImageLoader(config)
        self._face_detector = FaceDetector(config)
        self._image_quality = ImageQualityInspector(config)
        self._preprocessor = ModelPreprocessor(config)
        self._model_service = model_service

    def analyze(self, file_storage: FileStorage | None) -> dict:
        if file_storage is None:
            raise InvalidPayloadError("Envie a imagem no campo 'image' do formulário multipart.")

        filename = (file_storage.filename or "").strip()

        if not filename:
            raise InvalidPayloadError("Envie a imagem no campo 'image' do formulário multipart.")

        image_bytes = file_storage.read()

        loaded_image = self._image_loader.load(
            image_bytes=image_bytes,
            filename=filename,
        )

        face_detection = self._face_detector.ensure_single_face(loaded_image.image_bgr)
        quality = self._image_quality.inspect(loaded_image.image_bgr)
        tensor = self._preprocessor.transform(loaded_image.image_pil)
        prediction = self._model_service.predict(tensor)

        return {
            "filename": loaded_image.filename,
            "image": {
                "width": loaded_image.width,
                "height": loaded_image.height,
                "format": loaded_image.image_format,
            },
            "face_detection": {
                "faces_detected": face_detection.faces_detected,
                "bounding_boxes": face_detection.bounding_boxes,
            },
            "quality": quality,
            "prediction": prediction,
        }