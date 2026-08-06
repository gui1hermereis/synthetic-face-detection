from flask import Flask, jsonify
from werkzeug.exceptions import HTTPException

from .exceptions import ApiError

def register_error_handlers(app: Flask) -> None:
    @app.errorhandler(ApiError)
    def handle_api_error(error: ApiError):
        response = jsonify(
            {
                "error": {
                    "code": error.code,
                    "message": error.message,
                    "details": error.details,
                }
            }
        )
        response.status_code = error.status_code
        return response

    @app.errorhandler(HTTPException)
    def handle_http_exception(error: HTTPException):
        response = jsonify(
            {
                "error": {
                    "code": error.name.lower().replace(" ", "_"),
                    "message": error.description,
                    "details": {},
                }
            }
        )
        response.status_code = error.code or 500
        return response

    @app.errorhandler(Exception)
    def handle_unexpected_error(error: Exception):
        response = jsonify(
            {
                "error": {
                    "code": "internal_server_error",
                    "message": "Ocorreu um erro inesperado durante o processamento.",
                    "details": {"reason": str(error)},
                }
            }
        )
        response.status_code = 500
        return response