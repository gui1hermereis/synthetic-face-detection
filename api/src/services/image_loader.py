from dataclasses import dataclass
from io import BytesIO

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from ..errors.exceptions import ImageValidationError

@dataclass
class LoadedImage:
    filename: str
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
            raise ImageValidationError("O arquivo enviado está vazio.")

        if len(image_bytes) > self._max_content_length:
            raise ImageValidationError(
                "O arquivo enviado excede o tamanho máximo permitido.",
                details={"max_bytes": self._max_content_length},
            )

        try:
            with Image.open(BytesIO(image_bytes)) as image:
                image_format = image.format or "unknown"
                image = ImageOps.exif_transpose(image)
                image_rgb_pil = image.convert("RGB")
        except (UnidentifiedImageError, OSError) as error:
            raise ImageValidationError("O arquivo enviado não é uma imagem válida.") from error

        width, height = image_rgb_pil.size

        if width < self._min_width or height < self._min_height:
            raise ImageValidationError(
                "A imagem enviada possui resolução insuficiente para análise.",
                details={
                    "width": width,
                    "height": height,
                    "minimum_width": self._min_width,
                    "minimum_height": self._min_height,
                },
            )

        image_rgb = np.asarray(image_rgb_pil)
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        return LoadedImage(
            filename=filename,
            image_bgr=image_bgr,
            width=width,
            height=height,
            image_format=image_format,
        )
