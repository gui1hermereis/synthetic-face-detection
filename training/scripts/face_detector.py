from pathlib import Path

import cv2


CAMINHO_PROJETO = Path(__file__).resolve().parents[2]

CAMINHO_YUNET = (
    CAMINHO_PROJETO
    / "models"
    / "face_detection_yunet_2023mar.onnx"
)


class FaceDetector:
    def __init__(
        self,
        model_path=CAMINHO_YUNET,
        detector_max_size=320,
        score_threshold=0.9,
        nms_threshold=0.3,
        top_k=5000,
    ):
        model_path = Path(model_path)

        if not model_path.is_file():
            raise FileNotFoundError(
                f"Modelo YuNet não encontrado: {model_path}"
            )

        self.detector_max_size = detector_max_size

        self.detector = cv2.FaceDetectorYN.create(
            str(model_path),
            "",
            (detector_max_size, detector_max_size),
            score_threshold,
            nms_threshold,
            top_k,
        )

    def detect(self, image):
        if image is None:
            raise ValueError("Imagem inválida.")

        original_height, original_width = image.shape[:2]

        maior_dimensao = max(
            original_width,
            original_height,
        )

        scale = min(
            1.0,
            self.detector_max_size / maior_dimensao,
        )

        resized_width = int(
            round(original_width * scale)
        )

        resized_height = int(
            round(original_height * scale)
        )

        if scale < 1.0:
            image_detector = cv2.resize(
                image,
                (resized_width, resized_height),
                interpolation=cv2.INTER_AREA,
            )
        else:
            image_detector = image

        self.detector.setInputSize(
            (
                image_detector.shape[1],
                image_detector.shape[0],
            )
        )

        _, faces = self.detector.detect(
            image_detector
        )

        if faces is None:
            return []

        faces = faces.copy()

        if scale != 1.0:
            # x, y, w, h + 5 landmarks
            faces[:, :14] /= scale

        return faces

    def get_face(self, image):
        faces = self.detect(image)

        if len(faces) == 0:
            return None, "no_face"

        if len(faces) > 1:
            return None, "multiple_faces"

        x, y, w, h = faces[0][:4]

        bbox = (
            int(round(x)),
            int(round(y)),
            int(round(w)),
            int(round(h)),
        )

        return bbox, "valid"

    def crop_face(self, image, margin=0.20):
        bbox, status = self.get_face(image)

        if status != "valid":
            return None, status

        x, y, w, h = bbox

        image_height, image_width = image.shape[:2]

        cx = x + (w / 2)
        cy = y + (h / 2)

        side = max(w, h)

        side *= 1 + (2 * margin)

        side = int(round(side))

        side = min(
            side,
            image_width,
            image_height,
        )

        x1 = int(round(cx - side / 2))
        y1 = int(round(cy - side / 2))

        x1 = max(
            0,
            min(x1, image_width - side),
        )

        y1 = max(
            0,
            min(y1, image_height - side),
        )

        x2 = x1 + side
        y2 = y1 + side

        crop = image[
            y1:y2,
            x1:x2
        ]

        return crop, "valid"