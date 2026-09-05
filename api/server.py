from flask import Flask
from flask_cors import CORS

from src.config.settings import Settings
from src.errors.handlers import register_error_handlers
from src.routes import register_routes
from src.services.face_detector import FaceDetector

def create_app(test_config: dict | None = None) -> Flask:
    app = Flask(__name__)
    app.config.from_object(Settings())

    if test_config:
        app.config.update(test_config)

    # O modelo YuNet e carregado uma unica vez e reutilizado em cada requisicao.
    app.extensions["face_detector"] = FaceDetector(app.config)

    CORS(app)

    register_routes(app)
    register_error_handlers(app)

    return app
