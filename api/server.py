from flask import Flask
from flask_cors import CORS

from src.config.settings import Settings
from src.errors.handlers import register_error_handlers
from src.routes import register_routes
from src.services.face_detector import FaceDetector
from src.services.model_service import ModelService


def create_app(test_config: dict | None = None) -> Flask:
    app = Flask(__name__)
    app.config.from_object(Settings())

    if test_config:
        app.config.update(test_config)

    # Serviço responsável pelo modelo classificador.
    # A instância é criada uma única vez e reutilizada
    # durante toda a execução da aplicação.
    model_service = ModelService(app.config)

    # O FaceDetector utiliza os mesmos metadados de
    # pré-processamento salvos no artefato treinado.
    face_detector = FaceDetector(
        app.config,
        model_service.metadata,
    )

    app.extensions["model_service"] = model_service
    app.extensions["face_detector"] = face_detector

    CORS(app)

    register_routes(app)
    register_error_handlers(app)

    return app