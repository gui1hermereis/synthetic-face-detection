from flask import Flask
from flask_cors import CORS

from src.config.settings import Settings
from src.errors.handlers import register_error_handlers
from src.routes import register_routes

def create_app(test_config: dict | None = None) -> Flask:
    app = Flask(__name__)
    app.config.from_object(Settings())

    if test_config:
        app.config.update(test_config)

    CORS(app)

    register_routes(app)
    register_error_handlers(app)

    return app