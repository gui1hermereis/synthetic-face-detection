from flask import Blueprint, current_app, jsonify

health_blueprint = Blueprint("health", __name__)

@health_blueprint.get("/health")
def health_check():
    return jsonify(
        {
            "status": "ok",
            "message": "API Flask inicializada com sucesso.",
            "model_weights_path": current_app.config["MODEL_WEIGHTS_PATH"],
        }
    )
