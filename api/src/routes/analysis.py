from flask import Blueprint, current_app, jsonify, request

from ..security.api_key import require_api_key
from ..services.analysis_service import AnalysisService
from ..services.model_service import ModelService

analysis_blueprint = Blueprint("analysis", __name__)

@analysis_blueprint.post("/images/analyze")
@require_api_key
def analyze_image():
    file_storage = request.files.get("image")
    service = AnalysisService(current_app.config, ModelService(current_app.config))
    result = service.analyze(file_storage)
    return jsonify(result), 200
