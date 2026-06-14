"""
POST /api/analyze  — public route, no authentication required.
Accepts a plant leaf image, runs ML analysis, saves to MongoDB, returns result.
"""

import os
import uuid
from datetime import datetime, timezone

import requests
from flask import Blueprint, request, jsonify, current_app
from werkzeug.utils import secure_filename

from app.extensions import mongo
from app.models.analysis import build_document, serialize

analyze_bp = Blueprint("analyze", __name__)

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "gif", "webp"}


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def _call_ml_model(file_path: str, file_name: str) -> dict:
    ml_url = os.getenv("ML_MODEL_URL", "http://localhost:8000")
    try:
        with open(file_path, "rb") as f:
            response = requests.post(
                f"{ml_url}/predict",
                files={"file": (file_name, f)},
                timeout=30,
            )
        response.raise_for_status()
        return response.json()
    except Exception as exc:
        current_app.logger.warning("ML model unreachable (%s). Using local fallback.", exc)
        return _local_visual_fallback(file_path)


def _local_visual_fallback(file_path: str) -> dict:
    import sys
    ml_model_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "ml_model")
    )
    if ml_model_dir not in sys.path:
        sys.path.insert(0, ml_model_dir)
    try:
        from server import demo_predict
        with open(file_path, "rb") as f:
            image_bytes = f.read()
        return demo_predict(image_bytes)
    except Exception as inner_exc:
        current_app.logger.error("Local fallback failed: %s", inner_exc)
        return {
            "plantStatus": "Unknown",
            "diseaseName": "Unable to Identify",
            "confidence": 0.0,
            "waterStress": "Moderate",
            "recommendation": "Analysis service unavailable. Please try again.",
            "resourceOptimization": "",
            "visibleSymptoms": [],
            "predictionReason": "ML service offline.",
        }


@analyze_bp.route("/", methods=["POST"])
def analyze():
    if "image" not in request.files:
        return jsonify({"success": False, "message": "No image file provided"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"success": False, "message": "No file selected"}), 400

    if not allowed_file(file.filename):
        return jsonify({
            "success": False,
            "message": "Only image files (jpeg, jpg, png, gif, webp) are allowed",
        }), 400

    try:
        ext         = file.filename.rsplit(".", 1)[1].lower()
        unique_name = f"{int(datetime.now().timestamp() * 1000)}-{uuid.uuid4().hex[:8]}.{ext}"
        safe_name   = secure_filename(unique_name)
        upload_dir  = current_app.config["UPLOAD_FOLDER"]
        save_path   = os.path.join(upload_dir, safe_name)
        file.save(save_path)

        image_url  = f"/static/uploads/{safe_name}"
        prediction = _call_ml_model(save_path, safe_name)

        plant_status     = prediction.get("plantStatus")   or prediction.get("plant_status")   or "Unknown"
        disease_name     = prediction.get("diseaseName")   or prediction.get("disease_name")   or "None"
        confidence       = round(float(prediction.get("confidence", 0)), 1)
        water_stress     = prediction.get("waterStress")   or prediction.get("water_stress")   or "Low"
        recommendation   = prediction.get("recommendation")  or "No recommendation available."
        resource_opt     = prediction.get("resourceOptimization") or prediction.get("resource_optimization") or ""
        visible_symptoms = prediction.get("visibleSymptoms") or prediction.get("visible_symptoms") or []
        pred_reason      = prediction.get("predictionReason") or prediction.get("prediction_reason") or ""

        doc = build_document(
            image_url=image_url,
            image_name=file.filename,
            plant_status=plant_status,
            disease_name=disease_name,
            confidence=confidence,
            water_stress_level=water_stress,
            recommendation=recommendation,
            resource_optimization=resource_opt,
            visible_symptoms=visible_symptoms,
            prediction_reason=pred_reason,
        )
        result    = mongo.db.analyses.insert_one(doc)
        doc["_id"] = str(result.inserted_id)

        return jsonify({"success": True, "message": "Analysis complete", "data": serialize(doc)}), 200

    except Exception as exc:
        current_app.logger.error("Analysis error: %s", exc)
        return jsonify({"success": False, "message": "Analysis failed", "error": str(exc)}), 500
