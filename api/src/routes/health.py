from pathlib import Path

from flask import Blueprint, current_app, jsonify

health_blueprint = Blueprint("health", __name__)

@health_blueprint.get("/health")
def health_check():
    weights_path = Path(current_app.config["MODEL_WEIGHTS_PATH"])
    model_available = weights_path.is_file()

    return jsonify({
        "status": "ok" if model_available else "error",
        "message": "API Flask inicializada com sucesso.",
        "model_available": model_available,
        "model_weights": weights_path.name,
    }), 200 if model_available else 503
