from functools import wraps
import secrets

from flask import current_app, request

from ..errors.exceptions import AuthenticationError

def require_api_key(view_func):
    @wraps(view_func)
    def wrapped(*args, **kwargs):
        expected_api_key = current_app.config["API_KEY"]
        received_api_key = request.headers.get("X-API-Key", "")

        if not expected_api_key:
            raise AuthenticationError("A API key do servidor não foi configurada.")

        if not secrets.compare_digest(received_api_key, expected_api_key):
            raise AuthenticationError("API key inválida ou ausente.")

        return view_func(*args, **kwargs)

    return wrapped