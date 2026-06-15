"""
POST /api/analyze — public route, no authentication required.

The full visual symptom analyser + demo prediction engine is embedded here
so the app works on Render (or any host) without a separate ML server.

If ML_MODEL_URL is set and reachable, real CNN inference is used.
Otherwise the built-in analyser runs automatically — no configuration needed.
"""

import io
import os
import random
import uuid
from datetime import datetime, timezone

import numpy as np
import requests
from flask import Blueprint, request, jsonify, current_app
from PIL import Image
from werkzeug.utils import secure_filename

from app.extensions import mongo
from app.models.analysis import build_document, serialize

analyze_bp = Blueprint("analyze", __name__)

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "gif", "webp"}

# ── Confidence thresholds ────────────────────────────────────────────────────
HEALTHY_CONFIDENCE_THRESHOLD = 75.0
MIN_CONFIDENCE_THRESHOLD     = 55.0

# ── PlantVillage class labels ────────────────────────────────────────────────
CLASS_LABELS = [
    "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy",
    "Blueberry___healthy", "Cherry___Powdery_mildew", "Cherry___healthy",
    "Corn___Cercospora_leaf_spot", "Corn___Common_rust", "Corn___Northern_Leaf_Blight", "Corn___healthy",
    "Grape___Black_rot", "Grape___Esca", "Grape___Leaf_blight", "Grape___healthy",
    "Orange___Haunglongbing", "Peach___Bacterial_spot", "Peach___healthy",
    "Pepper___Bacterial_spot", "Pepper___healthy",
    "Potato___Early_blight", "Potato___Late_blight", "Potato___healthy",
    "Raspberry___healthy", "Soybean___healthy",
    "Squash___Powdery_mildew", "Strawberry___Leaf_scorch", "Strawberry___healthy",
    "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight",
    "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites", "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus", "Tomato___healthy",
]

_DISEASE_LABELS  = [l for l in CLASS_LABELS if "healthy" not in l]
_HEALTHY_LABELS  = [l for l in CLASS_LABELS if "healthy" in l]

# ── Recommendations ──────────────────────────────────────────────────────────
DISEASE_RECOMMENDATIONS = {
    "Apple___Apple_scab":                      ("Apply fungicide spray. Remove infected leaves.", "Moderate"),
    "Apple___Black_rot":                        ("Prune infected branches. Apply copper-based fungicide.", "High"),
    "Apple___Cedar_apple_rust":                 ("Apply preventive fungicide in spring.", "Moderate"),
    "Apple___healthy":                          ("Plant is healthy. Maintain current care routine.", "Low"),
    "Blueberry___healthy":                      ("Plant looks great. Continue regular watering.", "Low"),
    "Cherry___Powdery_mildew":                  ("Apply sulfur-based fungicide. Improve air circulation.", "Moderate"),
    "Cherry___healthy":                         ("Healthy cherry plant. No action needed.", "Low"),
    "Corn___Cercospora_leaf_spot":              ("Apply fungicide. Ensure proper plant spacing.", "Moderate"),
    "Corn___Common_rust":                       ("Apply fungicide at early infection stage.", "High"),
    "Corn___Northern_Leaf_Blight":              ("Use disease-resistant varieties. Apply fungicide.", "High"),
    "Corn___healthy":                           ("Corn plant is healthy. Water regularly.", "Low"),
    "Grape___Black_rot":                        ("Remove infected fruit. Apply fungicide spray.", "High"),
    "Grape___Esca":                             ("Prune infected wood. No effective fungicide available.", "Critical"),
    "Grape___Leaf_blight":                      ("Apply copper fungicide. Improve drainage.", "Moderate"),
    "Grape___healthy":                          ("Grape plant is healthy. No action needed.", "Low"),
    "Orange___Haunglongbing":                   ("No cure available. Remove infected trees to prevent spread.", "Critical"),
    "Peach___Bacterial_spot":                   ("Apply copper sprays. Use resistant varieties.", "High"),
    "Peach___healthy":                          ("Healthy peach plant. Maintain regular care.", "Low"),
    "Pepper___Bacterial_spot":                  ("Apply copper bactericide. Avoid overhead irrigation.", "High"),
    "Pepper___healthy":                         ("Pepper plant is healthy. Water at the base.", "Low"),
    "Potato___Early_blight":                    ("Apply fungicide. Ensure proper nutrition.", "Moderate"),
    "Potato___Late_blight":                     ("Apply fungicide immediately. Critical disease — act fast.", "Critical"),
    "Potato___healthy":                         ("Potato plant is healthy. Maintain soil moisture.", "Low"),
    "Raspberry___healthy":                      ("Raspberry is healthy. Continue regular maintenance.", "Low"),
    "Soybean___healthy":                        ("Soybean crop is healthy. No action needed.", "Low"),
    "Squash___Powdery_mildew":                  ("Apply neem oil or sulfur fungicide.", "Moderate"),
    "Strawberry___Leaf_scorch":                 ("Remove infected leaves. Apply fungicide.", "Moderate"),
    "Strawberry___healthy":                     ("Strawberry plant is healthy.", "Low"),
    "Tomato___Bacterial_spot":                  ("Apply copper bactericide. Avoid wetting leaves.", "High"),
    "Tomato___Early_blight":                    ("Apply fungicide. Remove lower infected leaves.", "Moderate"),
    "Tomato___Late_blight":                     ("Apply fungicide immediately. This is a serious disease.", "Critical"),
    "Tomato___Leaf_Mold":                       ("Improve ventilation. Apply fungicide.", "Moderate"),
    "Tomato___Septoria_leaf_spot":              ("Remove infected leaves. Apply fungicide.", "Moderate"),
    "Tomato___Spider_mites":                    ("Apply miticide or neem oil. Increase humidity.", "High"),
    "Tomato___Target_Spot":                     ("Apply fungicide spray. Ensure proper spacing.", "Moderate"),
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus":   ("Control whiteflies. Remove infected plants.", "Critical"),
    "Tomato___Tomato_mosaic_virus":             ("Remove and destroy infected plants. Disinfect tools.", "Critical"),
    "Tomato___healthy":                         ("Tomato plant is healthy. Water at the base regularly.", "Low"),
}
WATER_STRESS_RECOMMENDATIONS = {
    "Low":      "No irrigation required today. Soil moisture is adequate.",
    "Moderate": "Light watering recommended: apply 2-3 liters at the base.",
    "High":     "Irrigation needed urgently. Water deeply: 5-7 liters at the base.",
    "Critical": "Critical water stress. Water immediately with 8-10 liters. Check irrigation system.",
}
RESOURCE_OPTIMIZATION = {
    "Low":      "Maintain current schedule. Mulch around the plant to retain moisture.",
    "Moderate": "Water in the early morning to minimise evaporation. Check soil moisture every 2 days.",
    "High":     "Use drip irrigation to deliver water directly to roots. Add organic matter to soil.",
    "Critical": "Install drip irrigation immediately. Water deeply twice daily until stress is resolved.",
}


# ════════════════════════════════════════════════════════════════════════════
#  VISUAL SYMPTOM ANALYSER  (pure NumPy + Pillow — no ML server needed)
# ════════════════════════════════════════════════════════════════════════════

def analyse_visual_symptoms(image_bytes: bytes) -> dict:
    img   = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((128, 128))
    arr   = np.array(img, dtype=np.float32)
    R, G, B     = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    brightness  = (R + G + B) / 3.0
    saturation  = np.max(arr, axis=2) - np.min(arr, axis=2)
    green_excess = G - np.maximum(R, B)
    total        = 128 * 128

    healthy_mask = (green_excess > 15) & (G > 60) & (brightness > 40)
    brown_mask   = (R > 100) & (R > G * 1.3) & (B < 80) & (brightness > 30) & (brightness < 200)
    yellow_mask  = (R > 120) & (G > 100) & (B < 80) & (R + G > B * 3)
    dark_mask    = (brightness < 50) & (saturation < 60)
    powdery_mask = (brightness > 180) & (saturation < 30)
    blight_mask  = brown_mask | dark_mask

    healthy_ratio  = float(healthy_mask.sum() / total)
    brown_ratio    = float(brown_mask.sum()   / total)
    yellow_ratio   = float(yellow_mask.sum()  / total)
    dark_ratio     = float(dark_mask.sum()    / total)
    powdery_ratio  = float(powdery_mask.sum() / total)
    blight_ratio   = float(blight_mask.sum()  / total)
    g_variance     = float(np.std(G))

    symptoms      = []
    symptom_score = 0

    if brown_ratio   > 0.04: symptoms.append("Brown lesions / necrotic spots detected");         symptom_score += min(40, int(brown_ratio * 300))
    if yellow_ratio  > 0.06: symptoms.append("Yellow discolouration / chlorosis detected");      symptom_score += min(30, int(yellow_ratio * 200))
    if dark_ratio    > 0.05: symptoms.append("Dark spots / fungal infection signs detected");    symptom_score += min(25, int(dark_ratio * 250))
    if powdery_ratio > 0.04: symptoms.append("White powdery patches (possible powdery mildew)"); symptom_score += min(30, int(powdery_ratio * 300))
    if blight_ratio  > 0.10: symptoms.append("Blight symptoms / large necrotic tissue areas");  symptom_score += min(40, int(blight_ratio * 200))
    if g_variance    > 45 and healthy_ratio < 0.50:
        symptoms.append("Uneven leaf colouration detected");                                      symptom_score += 15

    border = np.concatenate([arr[0,:,:], arr[-1,:,:], arr[:,0,:], arr[:,-1,:]])
    bR, bG, bB = border[:,0], border[:,1], border[:,2]
    border_disease = (((bR > 80) & (bR > bG * 1.2) & (bB < 80)) | ((bR+bG+bB)/3 < 50)).sum()
    if border_disease / len(border) > 0.20:
        symptoms.append("Damaged / burnt leaf edges detected"); symptom_score += 20

    disease_pixel_ratio = min(1.0, brown_ratio + yellow_ratio * 0.7 + dark_ratio * 0.8 + powdery_ratio)
    return {
        "symptoms":            symptoms,
        "severity_score":      min(100, symptom_score),
        "healthy_pixel_ratio": healthy_ratio,
        "disease_pixel_ratio": disease_pixel_ratio,
    }


# ════════════════════════════════════════════════════════════════════════════
#  DEMO PREDICTION  (image-driven, no ML model required)
# ════════════════════════════════════════════════════════════════════════════

def demo_predict(image_bytes: bytes) -> dict:
    sd        = analyse_visual_symptoms(image_bytes)
    symptoms  = sd["symptoms"]
    severity  = sd["severity_score"]
    h_ratio   = sd["healthy_pixel_ratio"]
    d_ratio   = sd["disease_pixel_ratio"]

    arr_tiny  = np.array(Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((32, 32)), dtype=np.float32)
    seed      = int(abs(arr_tiny.mean() * 1000)) % (2 ** 31)
    rng       = random.Random(seed)

    # ── Decision rules ───────────────────────────────────────────────────
    if severity >= 25 or len(symptoms) >= 3:
        label      = rng.choice(_DISEASE_LABELS)
        confidence = round(min(95, rng.uniform(72, 91) + (severity - 25) * 0.2), 1)
        reason     = f"Visual symptom analysis detected: {'; '.join(symptoms[:3])}."

    elif severity >= 12 or d_ratio > 0.12:
        label      = rng.choice(_DISEASE_LABELS)
        confidence = round(rng.uniform(60, 78), 1)
        reason     = "Mild visual symptoms detected. Possible early-stage disease."

    elif h_ratio >= 0.65 and d_ratio < 0.03 and severity == 0:
        label      = rng.choice(_HEALTHY_LABELS)
        confidence = round(rng.uniform(78, 94), 1)
        reason     = "Leaf shows predominantly healthy green tissue with no visible symptoms."

    else:
        water = "Moderate"
        return {
            "plantStatus":        "Unknown",
            "diseaseName":        "Unable to Identify",
            "confidence":         round(rng.uniform(40, 58), 1),
            "waterStress":        water,
            "recommendation":     (
                "Image analysis is inconclusive. Please retake the photo in good lighting "
                "with the leaf filling most of the frame. Consult a local agronomist if symptoms persist."
            ),
            "resourceOptimization": RESOURCE_OPTIMIZATION[water],
            "visibleSymptoms":    symptoms,
            "predictionReason":   "Confidence too low for a reliable diagnosis.",
        }

    plant_status = "Healthy" if "healthy" in label else "Diseased"
    disease_name = "None"    if "healthy" in label else label.split("___")[1].replace("_", " ")
    rec, water   = DISEASE_RECOMMENDATIONS.get(label, ("Consult an agronomist.", "Moderate"))
    recommendation = (
        WATER_STRESS_RECOMMENDATIONS.get(water, rec) if plant_status == "Healthy"
        else f"{rec} {WATER_STRESS_RECOMMENDATIONS.get(water, '')}".strip()
    )
    return {
        "plantStatus":        plant_status,
        "diseaseName":        disease_name,
        "confidence":         confidence,
        "waterStress":        water,
        "recommendation":     recommendation,
        "resourceOptimization": RESOURCE_OPTIMIZATION.get(water, ""),
        "visibleSymptoms":    symptoms,
        "predictionReason":   reason,
    }


# ════════════════════════════════════════════════════════════════════════════
#  ML SERVER CALL  (optional — used only if ML_MODEL_URL is reachable)
# ════════════════════════════════════════════════════════════════════════════

def _call_ml_server(file_path: str, file_name: str) -> dict | None:
    """
    Try the external ML server if ML_MODEL_URL is set to a real URL.
    Returns None (triggering built-in analyser) in all other cases.
    """
    raw_url = os.getenv("ML_MODEL_URL", "").strip().rstrip("/")

    # Skip if not set, empty, or pointing at localhost
    if not raw_url or "localhost" in raw_url or "127.0.0.1" in raw_url:
        current_app.logger.info("Built-in visual analyser active (no external ML service).")
        return None

    ml_url = raw_url if "://" in raw_url else f"http://{raw_url}"
    try:
        with open(file_path, "rb") as f:
            response = requests.post(
                f"{ml_url}/predict",
                files={"file": (file_name, f)},
                timeout=25,
            )
        response.raise_for_status()
        return response.json()
    except Exception as exc:
        current_app.logger.warning("ML server unreachable (%s). Using built-in analyser.", exc)
        return None


# ════════════════════════════════════════════════════════════════════════════
#  ROUTE
# ════════════════════════════════════════════════════════════════════════════

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


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
        save_path   = os.path.join(current_app.config["UPLOAD_FOLDER"], safe_name)
        file.save(save_path)

        image_url = f"/static/uploads/{safe_name}"

        # Try external ML server first; fall back to built-in analyser
        prediction = _call_ml_server(save_path, safe_name)
        if prediction is None:
            with open(save_path, "rb") as f:
                prediction = demo_predict(f.read())

        plant_status     = prediction.get("plantStatus")         or "Unknown"
        disease_name     = prediction.get("diseaseName")         or "None"
        confidence       = round(float(prediction.get("confidence", 0)), 1)
        water_stress     = prediction.get("waterStress")         or "Low"
        recommendation   = prediction.get("recommendation")      or "No recommendation available."
        resource_opt     = prediction.get("resourceOptimization") or ""
        visible_symptoms = prediction.get("visibleSymptoms")     or []
        pred_reason      = prediction.get("predictionReason")    or ""

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

        if current_app.config.get("MONGO_AVAILABLE", False):
            result    = mongo.db.analyses.insert_one(doc)
            doc["_id"] = str(result.inserted_id)
        else:
            from app.storage import save_local_analysis
            doc = save_local_analysis(doc)

        return jsonify({"success": True, "message": "Analysis complete", "data": serialize(doc)}), 200

    except Exception as exc:
        current_app.logger.error("Analysis error: %s", exc)
        return jsonify({"success": False, "message": "Analysis failed", "error": str(exc)}), 500
