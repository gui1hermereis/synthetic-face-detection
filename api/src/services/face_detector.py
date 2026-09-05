from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ..errors.exceptions import FaceValidationError


@dataclass
class FaceDetectionResult:
    faces_detected: int
    bounding_boxes: list[list[int]]

    def crop_single_face(self, image_bgr: np.ndarray, padding_ratio: float) -> np.ndarray:
        """Retorna o recorte da unica face detectada, com margem proporcional."""
        x, y, width, height = self.bounding_boxes[0]
        image_height, image_width = image_bgr.shape[:2]
        padding_x = round(width * padding_ratio)
        padding_y = round(height * padding_ratio)

        left = max(0, x - padding_x)
        top = max(0, y - padding_y)
        right = min(image_width, x + width + padding_x)
        bottom = min(image_height, y + height + padding_y)

        return image_bgr[top:bottom, left:right]


class FaceDetector:
    def __init__(self, config):
        model_path = Path(config["FACE_DETECTOR_MODEL_PATH"])
        if not model_path.is_file():
            raise RuntimeError(
                "Modelo YuNet nao encontrado. Baixe face_detection_yunet_2023mar.onnx "
                f"ou configure FACE_DETECTOR_MODEL_PATH. Caminho esperado: {model_path}"
            )

        self._detector = cv2.FaceDetectorYN.create(
            str(model_path),
            "",
            (320, 320),
            config["FACE_DETECTOR_SCORE_THRESHOLD"],
            0.3,
            5000,
        )
        self._min_face_size = config["MIN_FACE_SIZE"]

        if self._detector is None:
            raise RuntimeError("Nao foi possivel inicializar o detector facial YuNet.")

    def ensure_single_face(self, image_bgr: np.ndarray) -> FaceDetectionResult:
        image_height, image_width = image_bgr.shape[:2]
        self._detector.setInputSize((image_width, image_height))
        _, faces = self._detector.detect(image_bgr)

        valid_faces = [] if faces is None else [
            face for face in faces
            if face[2] >= self._min_face_size and face[3] >= self._min_face_size
        ]
        bounding_boxes = [
            self._clip_bounding_box(face[:4], image_width, image_height)
            for face in valid_faces
        ]
        faces_detected = len(bounding_boxes)

        if faces_detected == 0:
            raise FaceValidationError(
                "Nao foi detectado um rosto humano na imagem enviada.",
                details={"faces_detected": 0},
            )

        if faces_detected > 1:
            raise FaceValidationError(
                "A imagem enviada contem mais de um rosto humano.",
                details={
                    "faces_detected": faces_detected,
                    "bounding_boxes": bounding_boxes,
                },
            )

        return FaceDetectionResult(faces_detected=1, bounding_boxes=bounding_boxes)

    @staticmethod
    def _clip_bounding_box(
        bounding_box: np.ndarray,
        image_width: int,
        image_height: int,
    ) -> list[int]:
        x, y, width, height = bounding_box
        left = max(0, int(round(x)))
        top = max(0, int(round(y)))
        right = min(image_width, int(round(x + width)))
        bottom = min(image_height, int(round(y + height)))
        return [left, top, right - left, bottom - top]
