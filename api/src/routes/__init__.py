from flask import Flask

from .analysis import analysis_blueprint
from .health import health_blueprint

def register_routes(app: Flask) -> None:
    app.register_blueprint(health_blueprint, url_prefix="/api/v1")
    app.register_blueprint(analysis_blueprint, url_prefix="/api/v1")