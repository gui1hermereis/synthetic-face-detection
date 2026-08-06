class ApiError(Exception):
    def __init__(self, message: str, status_code: int, code: str, details: dict | None = None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.code = code
        self.details = details or {}

class AuthenticationError(ApiError):
    def __init__(self, message: str, details: dict | None = None):
        super().__init__(message, status_code=401, code="authentication_error", details=details)

class InvalidPayloadError(ApiError):
    def __init__(self, message: str, details: dict | None = None):
        super().__init__(message, status_code=400, code="invalid_payload", details=details)

class ImageValidationError(ApiError):
    def __init__(self, message: str, details: dict | None = None):
        super().__init__(message, status_code=422, code="image_validation_error", details=details)

class FaceValidationError(ApiError):
    def __init__(self, message: str, details: dict | None = None):
        super().__init__(message, status_code=422, code="face_validation_error", details=details)

class ImageQualityError(ApiError):
    def __init__(self, message: str, details: dict | None = None):
        super().__init__(message, status_code=422, code="image_quality_error", details=details)

class ModelInferenceError(ApiError):
    def __init__(self, message: str, details: dict | None = None):
        super().__init__(message, status_code=500, code="model_inference_error", details=details)