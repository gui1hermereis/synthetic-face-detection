from dataclasses import dataclass
from pathlib import Path
from threading import Lock

import cv2
import numpy as np

from ..errors.exceptions import FaceValidationError


@dataclass
class FaceDetectionResult:
    faces_detected: int
    bounding_boxes: list[list[int]]

    def crop_single_face(
        self,
        image_bgr: np.ndarray,
        padding_ratio: float,
    ) -> np.ndarray:
        """
        Retorna o crop quadrado da face utilizando a mesma
        estratégia usada durante o treinamento.
        """
        x, y, width, height = self.bounding_boxes[0]

        image_height, image_width = image_bgr.shape[:2]

        center_x = x + (width / 2)
        center_y = y + (height / 2)

        side = int(
            round(
                max(width, height)
                * (1 + 2 * padding_ratio)
            )
        )

        side = min(
            side,
            image_width,
            image_height,
        )

        left = int(
            round(center_x - side / 2)
        )

        top = int(
            round(center_y - side / 2)
        )

        left = max(
            0,
            min(
                left,
                image_width - side,
            ),
        )

        top = max(
            0,
            min(
                top,
                image_height - side,
            ),
        )

        return image_bgr[
            top:top + side,
            left:left + side,
        ]


class FaceDetector:
    def __init__(
        self,
        config,
        model_metadata: dict,
    ):
        face_crop = model_metadata.get("face_crop")

        if not isinstance(face_crop, dict):
            raise RuntimeError(
                "Os metadados de detecção facial não foram "
                "encontrados no artefato do modelo."
            )

        detector_name = face_crop.get("detector")

        score_threshold = face_crop.get(
            "score_threshold"
        )

        single_face_only = face_crop.get(
            "single_face_only"
        )

        crop_shape = face_crop.get(
            "crop_shape"
        )

        if score_threshold is None:
            raise RuntimeError(
                "O artefato do modelo não contém "
                "'face_crop.score_threshold'."
            )

        score_threshold = float(
            score_threshold
        )

        if not 0.0 <= score_threshold <= 1.0:
            raise RuntimeError(
                "O score_threshold do detector facial é inválido."
            )

        if single_face_only is not True:
            raise RuntimeError(
                "Este serviço espera um modelo configurado "
                "para analisar exatamente um rosto."
            )

        if crop_shape != "square":
            raise RuntimeError(
                "Este serviço suporta apenas crop facial quadrado."
            )

        model_path = Path(
            config["FACE_DETECTOR_MODEL_PATH"]
        )

        if not model_path.is_file():
            raise RuntimeError(
                "Modelo YuNet não encontrado. "
                "Configure FACE_DETECTOR_MODEL_PATH. "
                f"Caminho esperado: {model_path}"
            )

        # Confere se o arquivo configurado corresponde ao
        # detector declarado no artefato.
        if detector_name:
            expected_detector_name = str(detector_name).split()[-1]

            if model_path.name != expected_detector_name:
                raise RuntimeError(
                    "O detector facial configurado é diferente "
                    "do detector utilizado no treinamento. "
                    f"Esperado: {expected_detector_name}. "
                    f"Recebido: {model_path.name}."
                )

        self._detector_max_size = 320

        self._score_threshold = score_threshold

        self._single_face_only = single_face_only

        self._detector = cv2.FaceDetectorYN.create(
            str(model_path),
            "",
            (
                self._detector_max_size,
                self._detector_max_size,
            ),
            self._score_threshold,
            0.3,
            5000,
        )

        if self._detector is None:
            raise RuntimeError(
                "Não foi possível inicializar "
                "o detector facial YuNet."
            )

        # O FaceDetector é compartilhado entre as requisições.
        # O Lock impede que duas threads alterem o estado
        # interno do YuNet ao mesmo tempo.
        self._lock = Lock()

        self._min_face_size = config[
            "MIN_FACE_SIZE"
        ]

    def ensure_single_face(
        self,
        image_bgr: np.ndarray,
    ) -> FaceDetectionResult:
        if image_bgr is None or image_bgr.size == 0:
            raise FaceValidationError(
                "A imagem enviada é inválida ou está vazia.",
                details={
                    "faces_detected": 0,
                },
            )

        image_height, image_width = (
            image_bgr.shape[:2]
        )

        scale = min(
            1.0,
            self._detector_max_size
            / max(
                image_width,
                image_height,
            ),
        )

        if scale < 1.0:
            detector_width = max(
                1,
                int(
                    round(
                        image_width * scale
                    )
                ),
            )

            detector_height = max(
                1,
                int(
                    round(
                        image_height * scale
                    )
                ),
            )

            detector_image = cv2.resize(
                image_bgr,
                (
                    detector_width,
                    detector_height,
                ),
                interpolation=cv2.INTER_AREA,
            )

        else:
            detector_image = image_bgr

        try:
            # setInputSize altera o estado interno do detector.
            # Por isso, setInputSize e detect precisam ficar
            # dentro do mesmo Lock.
            with self._lock:
                self._detector.setInputSize(
                    (
                        detector_image.shape[1],
                        detector_image.shape[0],
                    )
                )

                _, faces = self._detector.detect(
                    detector_image
                )

        except Exception as exc:
            raise FaceValidationError(
                "Não foi possível executar "
                "a detecção facial.",
                details={
                    "error": str(exc),
                },
            ) from exc

        # O YuNet retorna:
        #
        # 4 valores de bounding box
        # 10 valores de landmarks
        # 1 score
        #
        # Somente os primeiros 14 valores representam
        # coordenadas e precisam voltar para a escala
        # da imagem original.
        if (
            faces is not None
            and scale != 1.0
        ):
            faces = faces.copy()
            faces[:, :14] /= scale

        if faces is None:
            valid_faces = []

        else:
            valid_faces = [
                face
                for face in faces
                if (
                    face[2] >= self._min_face_size
                    and face[3] >= self._min_face_size
                )
            ]

        bounding_boxes = [
            self._clip_bounding_box(
                face[:4],
                image_width,
                image_height,
            )
            for face in valid_faces
        ]

        # Remove caixas inválidas que eventualmente tenham
        # ficado sem largura ou altura após o clipping.
        bounding_boxes = [
            box
            for box in bounding_boxes
            if box[2] > 0 and box[3] > 0
        ]

        faces_detected = len(
            bounding_boxes
        )

        if faces_detected == 0:
            raise FaceValidationError(
                "Não foi detectado um rosto humano "
                "na imagem enviada.",
                details={
                    "faces_detected": 0,
                },
            )

        if (
            self._single_face_only
            and faces_detected > 1
        ):
            raise FaceValidationError(
                "A imagem enviada contém mais "
                "de um rosto humano.",
                details={
                    "faces_detected":
                        faces_detected,
                    "bounding_boxes":
                        bounding_boxes,
                },
            )

        return FaceDetectionResult(
            faces_detected=faces_detected,
            bounding_boxes=bounding_boxes,
        )

    @staticmethod
    def _clip_bounding_box(
        bounding_box: np.ndarray,
        image_width: int,
        image_height: int,
    ) -> list[int]:
        x, y, width, height = bounding_box

        left = max(
            0,
            int(round(x)),
        )

        top = max(
            0,
            int(round(y)),
        )

        right = min(
            image_width,
            int(round(x + width)),
        )

        bottom = min(
            image_height,
            int(round(y + height)),
        )

        return [
            left,
            top,
            max(0, right - left),
            max(0, bottom - top),
        ]