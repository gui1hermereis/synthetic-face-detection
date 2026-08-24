from dataclasses import dataclass

import cv2
import numpy as np

from ..errors.exceptions import FaceValidationError


@dataclass
class FaceDetectionResult:
    faces_detected: int
    bounding_boxes: list[list[int]]

class FaceDetector:
    def __init__(self, config):
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._classifier = cv2.CascadeClassifier(cascade_path)
        self._min_face_size = config["MIN_FACE_SIZE"]

        if self._classifier.empty():
            raise RuntimeError("Não foi possível carregar o classificador de detecção facial.")

    def ensure_single_face(self, image_bgr: np.ndarray) -> FaceDetectionResult:
        grayscale = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

        faces = self._classifier.detectMultiScale(
            grayscale,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(self._min_face_size, self._min_face_size),
        )

        faces_detected = len(faces)
        bounding_boxes = [[int(x), int(y), int(w), int(h)] for x, y, w, h in faces]

        if faces_detected == 0:
            raise FaceValidationError(
                "A imagem enviada não contém um rosto humano detectável.",
                details={"faces_detected": 0},
            )

        if faces_detected > 1:
            raise FaceValidationError(
                "A imagem enviada contém mais de um rosto humano.",
                details={
                    "faces_detected": faces_detected,
                    "bounding_boxes": bounding_boxes,
                },
            )

        return FaceDetectionResult(
            faces_detected=1,
            bounding_boxes=bounding_boxes,
        )