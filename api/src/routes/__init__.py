from flask import Flask

from .analysis import analysis_blueprint
from .health import health_blueprint

URL_PREFIX = "/api/v1"

def register_routes(app: Flask) -> None:
    app.register_blueprint(health_blueprint, url_prefix=URL_PREFIX)
    app.register_blueprint(analysis_blueprint, url_prefix=URL_PREFIX)