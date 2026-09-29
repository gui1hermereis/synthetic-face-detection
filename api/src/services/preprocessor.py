import torch
from PIL import Image
from torchvision import transforms

class ModelPreprocessor:
    def __init__(self, metadata: dict):
        input_size = metadata["image_size"]
        normalization = metadata["normalization"]

        self._transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=normalization["mean"],
                std=normalization["std"],
            ),
        ])

    def transform(self, image: Image.Image) -> torch.Tensor:
        image = image.convert("RGB")
        return self._transform(image).unsqueeze(0)
