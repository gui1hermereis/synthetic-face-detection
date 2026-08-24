import torch
from PIL import Image
from torchvision import transforms

class ModelPreprocessor:
    def __init__(self, config):
        input_size = config["MODEL_INPUT_SIZE"]

        self._transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def transform(self, image: Image.Image) -> torch.Tensor:
        image = image.convert("RGB")
        return self._transform(image).unsqueeze(0)