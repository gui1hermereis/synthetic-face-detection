from flask import Blueprint, current_app, jsonify, request

from ..security.api_key import require_api_key
from ..services.analysis_service import AnalysisService
from ..services.model_service import ModelService

analysis_blueprint = Blueprint("analysis", __name__)

@analysis_blueprint.post("/images/analyze")
@require_api_key
def analyze_image():
    file_storage = request.files.get("image")

    model_service = ModelService(current_app.config)
    analysis_service = AnalysisService(
        current_app.config,
        model_service,
        current_app.extensions["face_detector"],
    )

    result = analysis_service.analyze(file_storage)

    return jsonify(result), 200
