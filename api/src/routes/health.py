from flask import Blueprint, current_app, jsonify

health_blueprint = Blueprint("health", __name__)


@health_blueprint.get("/health")
def health_check():
    metadata = current_app.extensions["model_service"].metadata
    return jsonify({
        "status": "ok",
        "message": "API Flask inicializada com sucesso.",
        "model_available": True,
        "model": {
            "architecture": metadata["architecture"],
            "input_size": metadata["image_size"],
            "input_channels": 3,
            "class_names": metadata["class_names"],
        },
    }), 200
