from dataclasses import dataclass

import cv2
import numpy as np

from ..errors.exceptions import ImageQualityError

@dataclass
class ImageQualityMetrics:
    blur_score: float
    contrast_score: float
    is_blurry: bool
    has_low_contrast: bool

class ImageQualityInspector:
    def __init__(self, config):
        self._blur_threshold = config["BLUR_THRESHOLD"]
        self._contrast_threshold = config["CONTRAST_THRESHOLD"]

    def inspect(self, image_bgr: np.ndarray) -> dict:
        grayscale = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        blur_score = float(cv2.Laplacian(grayscale, cv2.CV_64F).var())
        contrast_score = float(grayscale.std())

        metrics = ImageQualityMetrics(
            blur_score=blur_score,
            contrast_score=contrast_score,
            is_blurry=blur_score < self._blur_threshold,
            has_low_contrast=contrast_score < self._contrast_threshold,
        )

        if metrics.is_blurry:
            raise ImageQualityError(
                "A imagem enviada esta muito tremida ou desfocada para analise confiavel.",
                details={"blur_score": metrics.blur_score, "required_minimum": self._blur_threshold},
            )

        if metrics.has_low_contrast:
            raise ImageQualityError(
                "A imagem enviada esta com qualidade muito baixa para analise confiavel.",
                details={"contrast_score": metrics.contrast_score, "required_minimum": self._contrast_threshold},
            )

        return {
            "blur_score": round(metrics.blur_score, 4),
            "contrast_score": round(metrics.contrast_score, 4),
            "blur_threshold": self._blur_threshold,
            "contrast_threshold": self._contrast_threshold,
        }
