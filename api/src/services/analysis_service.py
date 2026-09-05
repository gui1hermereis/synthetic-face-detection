import cv2
from PIL import Image
from werkzeug.datastructures import FileStorage

from ..errors.exceptions import InvalidPayloadError
from .face_detector import FaceDetector
from .image_loader import ImageLoader
from .image_quality import ImageQualityInspector
from .model_service import ModelService
from .preprocessor import ModelPreprocessor

class AnalysisService:
    def __init__(self, config, model_service: ModelService, face_detector: FaceDetector):
        self._image_loader = ImageLoader(config)
        self._face_detector = face_detector
        self._image_quality = ImageQualityInspector(config)
        self._preprocessor = ModelPreprocessor(config)
        self._model_service = model_service
        self._face_crop_padding_ratio = config["FACE_CROP_PADDING_RATIO"]

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
        face_image_bgr = face_detection.crop_single_face(
            loaded_image.image_bgr,
            self._face_crop_padding_ratio,
        )
        quality = self._image_quality.inspect(face_image_bgr)
        face_image_rgb = cv2.cvtColor(face_image_bgr, cv2.COLOR_BGR2RGB)
        tensor = self._preprocessor.transform(Image.fromarray(face_image_rgb))
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
