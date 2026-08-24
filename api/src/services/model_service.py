from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models

from ..errors.exceptions import ModelInferenceError

class ModelService:
    _model = None

    def __init__(self, config):
        self._config = config
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._weights_path = Path(self._config["MODEL_WEIGHTS_PATH"])

    def _build_model(self) -> nn.Module:
        class_names = self._config["MODEL_CLASS_NAMES"]

        if not self._weights_path.is_file():
            raise ModelInferenceError(
                "Arquivo de pesos do modelo não encontrado.",
                details={"weights_path": str(self._weights_path)},
            )

        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, len(class_names))

        try:
            state_dict = torch.load(
                self._weights_path,
                map_location=self._device,
                weights_only=True,
            )
            model.load_state_dict(state_dict)
        except Exception as exc:
            raise ModelInferenceError(
                "Não foi possível carregar os pesos do modelo.",
                details={
                    "weights_path": str(self._weights_path),
                    "error": str(exc),
                },
            ) from exc

        model.to(self._device)
        model.eval()

        return model

    def _get_model(self) -> nn.Module:
        if self.__class__._model is None:
            self.__class__._model = self._build_model()

        return self.__class__._model

    def _validate_tensor(self, tensor: torch.Tensor) -> None:
        input_size = self._config["MODEL_INPUT_SIZE"]

        if tensor.ndim != 4:
            raise ModelInferenceError(
                "Formato do tensor de entrada inválido.",
                details={
                    "expected": "[batch_size, 3, height, width]",
                    "received": list(tensor.shape),
                },
            )

        if tensor.shape[0] != 1:
            raise ModelInferenceError(
                "O modelo espera uma imagem por inferência.",
                details={
                    "expected_batch_size": 1,
                    "received_batch_size": tensor.shape[0],
                },
            )

        if tensor.shape[1] != 3:
            raise ModelInferenceError(
                "O modelo espera imagens RGB com 3 canais.",
                details={
                    "expected_channels": 3,
                    "received_channels": tensor.shape[1],
                },
            )

        if tensor.shape[2] != input_size or tensor.shape[3] != input_size:
            raise ModelInferenceError(
                "Dimensões da imagem incompatíveis com o modelo.",
                details={
                    "expected_size": [input_size, input_size],
                    "received_size": [tensor.shape[2], tensor.shape[3]],
                },
            )

    def predict(self, tensor: torch.Tensor) -> dict:
        self._validate_tensor(tensor)

        model = self._get_model()
        class_names = self._config["MODEL_CLASS_NAMES"]

        tensor = tensor.to(self._device, non_blocking=True)

        try:
            with torch.inference_mode():
                logits = model(tensor)
                probabilities = torch.softmax(logits, dim=1)[0]
        except Exception as exc:
            raise ModelInferenceError(
                "Erro durante a inferência do modelo.",
                details={"error": str(exc)},
            ) from exc

        probabilities = probabilities.cpu().tolist()
        best_index = max(range(len(probabilities)), key=probabilities.__getitem__)

        return {
            "label": class_names[best_index],
            "confidence": round(float(probabilities[best_index]), 6),
            "probabilities": {
                class_names[index]: round(float(probability), 6)
                for index, probability in enumerate(probabilities)
            },
            "model": {
                "name": "resnet50",
                "input_size": self._config["MODEL_INPUT_SIZE"],
                "device": str(self._device),
                "weights": self._weights_path.name,
            },
        }