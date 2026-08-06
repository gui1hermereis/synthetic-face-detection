import json
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

    def _build_model(self) -> nn.Module:
        class_names = self._config["MODEL_CLASS_NAMES"]
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        model.fc = nn.Linear(model.fc.in_features, len(class_names))

        weights_path = Path(self._config["MODEL_WEIGHTS_PATH"])
        if not weights_path.exists():
            raise ModelInferenceError(
                "Arquivo de pesos do modelo nao encontrado.",
                details={"model_weights_path": str(weights_path)},
            )

        state_dict = torch.load(weights_path, map_location=self._device)
        model.load_state_dict(state_dict)
        model.to(self._device)
        model.eval()
        return model

    def _get_model(self) -> nn.Module:
        if self.__class__._model is None:
            self.__class__._model = self._build_model()
        return self.__class__._model

    def _load_metadata(self) -> dict | None:
        metadata_path = Path(self._config["MODEL_METADATA_PATH"])
        if not metadata_path.exists():
            return None
        try:
            return json.loads(metadata_path.read_text())
        except json.JSONDecodeError:
            return {"warning": "Nao foi possivel interpretar o arquivo de metricas do modelo."}

    def predict(self, tensor: torch.Tensor) -> dict:
        model = self._get_model()
        class_names = self._config["MODEL_CLASS_NAMES"]

        with torch.no_grad():
            logits = model(tensor.to(self._device))
            probabilities = torch.softmax(logits, dim=1)[0].cpu().tolist()

        best_index = max(range(len(probabilities)), key=probabilities.__getitem__)
        best_label = class_names[best_index]

        return {
            "label": best_label,
            "confidence": round(float(probabilities[best_index]), 6),
            "probabilities": {
                class_names[index]: round(float(probability), 6)
                for index, probability in enumerate(probabilities)
            },
            "model": {
                "name": "resnet50",
                "input_size": self._config["MODEL_INPUT_SIZE"],
                "device": str(self._device),
                "weights_path": self._config["MODEL_WEIGHTS_PATH"],
            },
            "training_metrics": self._load_metadata(),
        }