from dataclasses import dataclass
from io import BytesIO

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

from ..errors.exceptions import ImageValidationError

@dataclass
class LoadedImage:
    filename: str
    image_pil: Image.Image
    image_bgr: np.ndarray
    width: int
    height: int
    image_format: str

class ImageLoader:
    def __init__(self, config):
        self._min_width = config["MIN_IMAGE_WIDTH"]
        self._min_height = config["MIN_IMAGE_HEIGHT"]
        self._max_content_length = config["MAX_CONTENT_LENGTH"]

    def load(self, image_bytes: bytes, filename: str) -> LoadedImage:
        if not image_bytes:
            raise ImageValidationError("O arquivo enviado esta vazio.")

        if len(image_bytes) > self._max_content_length:
            raise ImageValidationError(
                "O arquivo enviado excede o tamanho maximo permitido.",
                details={"max_bytes": self._max_content_length},
            )

        try:
            image_file = BytesIO(image_bytes)
            image_pil = Image.open(image_file)
            image_format = image_pil.format or "unknown"
            image_pil = image_pil.convert("RGB")
        except UnidentifiedImageError as error:
            raise ImageValidationError("O arquivo enviado nao e uma imagem valida.") from error

        image_rgb = np.array(image_pil)
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        width, height = image_pil.size

        if width < self._min_width or height < self._min_height:
            raise ImageValidationError(
                "A imagem enviada possui resolucao insuficiente para analise.",
                details={
                    "width": width,
                    "height": height,
                    "minimum_width": self._min_width,
                    "minimum_height": self._min_height,
                },
            )

        return LoadedImage(
            filename=filename,
            image_pil=image_pil,
            image_bgr=image_bgr,
            width=width,
            height=height,
            image_format=image_format,
        )