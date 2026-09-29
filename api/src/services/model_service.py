from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models

from ..errors.exceptions import ModelInferenceError

class ModelService:
    def __init__(self, config):
        self._config = config
        self._device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self._weights_path = Path(
            self._config["MODEL_WEIGHTS_PATH"]
        )

        self._artifact = None
        self._metadata = None
        self._model = None

    @property
    def metadata(self) -> dict:
        """
        Metadados do artefato treinado.

        São carregados somente quando necessários.
        """
        if self._metadata is None:
            self._artifact = self._load_artifact()
            self._metadata = self._metadata_from_artifact()

        return self._metadata

    def _load_artifact(self) -> dict:
        if not self._weights_path.is_file():
            raise ModelInferenceError(
                "Arquivo de pesos do modelo não encontrado.",
                details={
                    "weights_path": str(self._weights_path),
                },
            )

        try:
            loaded = torch.load(
                self._weights_path,
                map_location="cpu",
                weights_only=True,
            )

        except Exception as exc:
            raise ModelInferenceError(
                "Não foi possível ler o artefato do modelo.",
                details={
                    "weights_path": str(self._weights_path),
                    "error": str(exc),
                },
            ) from exc

        # Compatibilidade com checkpoints antigos que
        # continham somente o state_dict.
        if isinstance(loaded, dict) and "state_dict" in loaded:
            return loaded

        return {
            "state_dict": loaded,
        }

    def _metadata_from_artifact(self) -> dict:
        normalization = self._artifact.get(
            "normalization",
            {
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            },
        )

        class_names = self._artifact.get(
            "class_names",
            self._config["MODEL_CLASS_NAMES"],
        )

        class_names = list(class_names)

        class_to_index = self._artifact.get(
            "class_to_index",
            {
                name: index
                for index, name in enumerate(class_names)
            },
        )

        # Garante que o modelo é exatamente binário:
        # fake / real.
        if set(class_names) != {"fake", "real"}:
            raise ModelInferenceError(
                "O modelo deve declarar exatamente as classes "
                "'fake' e 'real'.",
                details={
                    "class_names": class_names,
                },
            )

        if set(class_to_index.keys()) != {"fake", "real"}:
            raise ModelInferenceError(
                "O mapeamento de classes do modelo é inválido.",
                details={
                    "class_to_index": class_to_index,
                },
            )

        # Confere se class_names e class_to_index representam
        # exatamente a mesma ordem utilizada no treinamento.
        for class_name, index in class_to_index.items():
            if (
                not isinstance(index, int)
                or index < 0
                or index >= len(class_names)
                or class_names[index] != class_name
            ):
                raise ModelInferenceError(
                    "Os metadados class_names e class_to_index "
                    "são incompatíveis.",
                    details={
                        "class_names": class_names,
                        "class_to_index": class_to_index,
                    },
                )

        threshold = float(
            self._artifact.get(
                "fake_probability_threshold",
                0.5,
            )
        )

        if not 0.0 <= threshold <= 1.0:
            raise ModelInferenceError(
                "O limiar de probabilidade de fake é inválido.",
                details={
                    "fake_probability_threshold": threshold,
                },
            )

        return {
            "architecture": self._artifact.get(
                "architecture",
                self._config["MODEL_ARCHITECTURE"],
            ),

            "image_size": int(
                self._artifact.get(
                    "image_size",
                    self._config["MODEL_INPUT_SIZE"],
                )
            ),

            "class_names": class_names,

            "class_to_index": class_to_index,

            "normalization": normalization,

            "fake_probability_threshold": threshold,

            "threshold_selection": self._artifact.get(
                "threshold_selection"
            ),

            "face_crop": self._artifact.get(
                "face_crop"
            ),
        }

    def _build_model(self) -> nn.Module:
        metadata = self.metadata

        architecture = metadata["architecture"]
        class_names = metadata["class_names"]

        builders = {
            "resnet18": models.resnet18,
            "resnet34": models.resnet34,
            "resnet50": models.resnet50,
        }

        if architecture not in builders:
            raise ModelInferenceError(
                "Arquitetura de modelo não suportada.",
                details={
                    "architecture": architecture,
                },
            )

        model = builders[architecture](
            weights=None
        )

        model.fc = nn.Linear(
            model.fc.in_features,
            len(class_names),
        )

        try:
            model.load_state_dict(
                self._artifact["state_dict"]
            )

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
        if self._model is None:
            self._model = self._build_model()

        return self._model

    def _validate_tensor(
        self,
        tensor: torch.Tensor,
    ) -> None:
        input_size = self.metadata["image_size"]

        if tensor.ndim != 4:
            raise ModelInferenceError(
                "Formato do tensor de entrada inválido.",
                details={
                    "expected": (
                        "[batch_size, 3, height, width]"
                    ),
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

        if (
            tensor.shape[2] != input_size
            or tensor.shape[3] != input_size
        ):
            raise ModelInferenceError(
                "Dimensões da imagem incompatíveis com o modelo.",
                details={
                    "expected_size": [
                        input_size,
                        input_size,
                    ],
                    "received_size": [
                        tensor.shape[2],
                        tensor.shape[3],
                    ],
                },
            )

    def predict(
        self,
        tensor: torch.Tensor,
    ) -> dict:
        self._validate_tensor(tensor)

        model = self._get_model()

        metadata = self.metadata
        class_names = metadata["class_names"]
        class_to_index = metadata["class_to_index"]

        fake_index = class_to_index["fake"]
        real_index = class_to_index["real"]

        threshold = metadata[
            "fake_probability_threshold"
        ]

        tensor = tensor.to(
            self._device,
            non_blocking=True,
        )

        try:
            with torch.inference_mode():
                logits = model(tensor)

                probabilities_tensor = torch.softmax(
                    logits,
                    dim=1,
                )[0]

        except Exception as exc:
            raise ModelInferenceError(
                "Erro durante a inferência do modelo.",
                details={
                    "error": str(exc),
                },
            ) from exc

        probabilities = (
            probabilities_tensor
            .cpu()
            .tolist()
        )

        fake_probability = float(
            probabilities[fake_index]
        )

        real_probability = float(
            probabilities[real_index]
        )

        # A decisão NÃO utiliza argmax.
        #
        # Ela utiliza exatamente o threshold escolhido
        # na validação durante o treinamento.
        #
        # Exemplo:
        # threshold = 0.20
        #
        # P(fake) >= 0.20 -> fake
        # P(fake) <  0.20 -> real
        if fake_probability >= threshold:
            predicted_index = fake_index
        else:
            predicted_index = real_index

        predicted_label = class_names[
            predicted_index
        ]

        return {
            "label": predicted_label,

            # Mantido para compatibilidade com clientes
            # que já consumiam este campo.
            "confidence": round(
                float(
                    probabilities[predicted_index]
                ),
                6,
            ),

            # Mais importante para interpretar a decisão
            # com threshold diferente de 0.5.
            "probability_fake": round(
                fake_probability,
                6,
            ),

            "probability_real": round(
                real_probability,
                6,
            ),

            "decision_threshold": round(
                threshold,
                6,
            ),

            "probabilities": {
                class_names[index]: round(
                    float(probability),
                    6,
                )
                for index, probability
                in enumerate(probabilities)
            },

            "decision": {
                "positive_class": "fake",
                "rule": "probability_fake >= threshold",
                "threshold": round(
                    threshold,
                    6,
                ),
                "threshold_selection":
                    metadata["threshold_selection"],
            },

            "model": {
                "name": metadata["architecture"],
                "input_size": metadata["image_size"],
                "device": str(self._device),
                "weights": self._weights_path.name,
            },
        }