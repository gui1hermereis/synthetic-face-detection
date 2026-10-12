import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

class ModelPreprocessor:
    def __init__(self, metadata: dict):
        self._input_size = metadata["image_size"]
        normalization = metadata["normalization"]

        self._transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=normalization["mean"],
                std=normalization["std"],
            ),
        ])

    def transform(self, image: Image.Image) -> torch.Tensor:
        image_rgb = np.asarray(image.convert("RGB"))

        image_resized = cv2.resize(
            image_rgb,
            (self._input_size, self._input_size),
            interpolation=cv2.INTER_AREA,
        )

        return self._transform(image_resized).unsqueeze(0)