class ApiError(Exception):
    def __init__(self, message: str, status_code: int, code: str, details: dict | None = None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.code = code
        self.details = details or {}

class AuthenticationError(ApiError):
    def __init__(self, message: str, details: dict | None = None):
        super().__init__(message, 401, "authentication_error", details)

class InvalidPayloadError(ApiError):
    def __init__(self, message: str, details: dict | None = None):
        super().__init__(message, 400, "invalid_payload", details)

class ImageValidationError(ApiError):
    def __init__(self, message: str, details: dict | None = None):
        super().__init__(message, 422, "image_validation_error", details)

class FaceValidationError(ApiError):
    def __init__(self, message: str, details: dict | None = None):
        super().__init__(message, 422, "face_validation_error", details)

class ImageQualityError(ApiError):
    def __init__(self, message: str, details: dict | None = None):
        super().__init__(message, 422, "image_quality_error", details)

class ModelUnavailableError(ApiError):
    def __init__(self, message: str, details: dict | None = None):
        super().__init__(message, 503, "model_unavailable", details)

class ModelInferenceError(ApiError):
    def __init__(self, message: str, details: dict | None = None):
        super().__init__(message, 500, "model_inference_error", details)