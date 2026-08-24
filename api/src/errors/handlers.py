from flask import Flask, jsonify
from werkzeug.exceptions import HTTPException

from .exceptions import ApiError

def register_error_handlers(app: Flask) -> None:
    @app.errorhandler(ApiError)
    def handle_api_error(error: ApiError):
        return jsonify({
            "error": {
                "code": error.code,
                "message": error.message,
                "details": error.details,
            }
        }), error.status_code

    @app.errorhandler(HTTPException)
    def handle_http_exception(error: HTTPException):
        return jsonify({
            "error": {
                "code": error.name.lower().replace(" ", "_"),
                "message": error.description,
                "details": {},
            }
        }), error.code or 500

    @app.errorhandler(Exception)
    def handle_unexpected_error(error: Exception):
        app.logger.exception("Erro inesperado durante o processamento.")

        return jsonify({
            "error": {
                "code": "internal_server_error",
                "message": "Ocorreu um erro inesperado durante o processamento.",
                "details": {},
            }
        }), 500