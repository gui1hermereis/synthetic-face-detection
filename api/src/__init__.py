from flask import Flask

from .config.settings import Settings
from .errors.handlers import register_error_handlers
from .routes import register_routes

def create_app(test_config: dict | None = None) -> Flask:
    app = Flask(__name__)
    app.config.from_object(Settings())

    if test_config:
        app.config.update(test_config)

    register_routes(app)
    register_error_handlers(app)

    return app
