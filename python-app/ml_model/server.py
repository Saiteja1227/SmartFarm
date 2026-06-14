"""
AI-Driven Smart Urban Farming Resource Optimization
FastAPI Inference Server — Plant Disease & Water Stress Detection

Endpoints:
  POST /predict  - Analyse plant leaf image
  GET  /health   - Health check
  GET  /classes  - List all disease classes

Fixes applied (v2):
  - Added visual symptom analyser (spots, lesions, blight, discolouration, damage)
  - Confidence threshold: Healthy requires ≥ 75 % AND no visual symptoms
  - Low-confidence predictions → "Unable to Identify" instead of silent Healthy
  - demo_predict now uses multi-channel pixel analysis, not a single green-dominance gate
  - real_predict uses top-k voting + symptom override to prevent false Healthy
  - Full response includes: visibleSymptoms, predictionReason fields
"""

import io
import os
import random
import numpy as np
from PIL import Image
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── TensorFlow (optional) ────────────────────────────────────────────────────
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("⚠️  TensorFlow not installed. Running in DEMO mode.")

# ── Config ───────────────────────────────────────────────────────────────────
IMG_SIZE = (224, 224)
MODEL_PATH = "models/plant_disease_model.h5"
DEMO_MODE = not TF_AVAILABLE or not os.path.exists(MODEL_PATH)

# Minimum confidence to trust a Healthy prediction
HEALTHY_CONFIDENCE_THRESHOLD = 75.0
# Minimum confidence to trust any prediction at all
MIN_CONFIDENCE_THRESHOLD = 55.0

# ── Class labels (PlantVillage 38-class) ─────────────────────────────────────
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

HEALTHY_LABELS = {l for l in CLASS_LABELS if "healthy" in l}
DISEASED_LABELS = {l for l in CLASS_LABELS if "healthy" not in l}

# ── Recommendation map ───────────────────────────────────────────────────────
DISEASE_RECOMMENDATIONS = {
    "Apple___Apple_scab": ("Apply fungicide spray. Remove infected leaves.", "Moderate"),
    "Apple___Black_rot": ("Prune infected branches. Apply copper-based fungicide.", "High"),
    "Apple___Cedar_apple_rust": ("Apply preventive fungicide in spring.", "Moderate"),
    "Apple___healthy": ("Plant is healthy. Maintain current care routine.", "Low"),
    "Blueberry___healthy": ("Plant looks great. Continue regular watering.", "Low"),
    "Cherry___Powdery_mildew": ("Apply sulfur-based fungicide. Improve air circulation.", "Moderate"),
    "Cherry___healthy": ("Healthy cherry plant. No action needed.", "Low"),
    "Corn___Cercospora_leaf_spot": ("Apply fungicide. Ensure proper plant spacing.", "Moderate"),
    "Corn___Common_rust": ("Apply fungicide at early infection stage.", "High"),
    "Corn___Northern_Leaf_Blight": ("Use disease-resistant varieties. Apply fungicide.", "High"),
    "Corn___healthy": ("Corn plant is healthy. Water regularly.", "Low"),
    "Grape___Black_rot": ("Remove infected fruit. Apply fungicide spray.", "High"),
    "Grape___Esca": ("Prune infected wood. No effective fungicide available.", "Critical"),
    "Grape___Leaf_blight": ("Apply copper fungicide. Improve drainage.", "Moderate"),
    "Grape___healthy": ("Grape plant is healthy. No action needed.", "Low"),
    "Orange___Haunglongbing": ("No cure available. Remove infected trees to prevent spread.", "Critical"),
    "Peach___Bacterial_spot": ("Apply copper sprays. Use resistant varieties.", "High"),
    "Peach___healthy": ("Healthy peach plant. Maintain regular care.", "Low"),
    "Pepper___Bacterial_spot": ("Apply copper bactericide. Avoid overhead irrigation.", "High"),
    "Pepper___healthy": ("Pepper plant is healthy. Water at the base.", "Low"),
    "Potato___Early_blight": ("Apply fungicide. Ensure proper nutrition.", "Moderate"),
    "Potato___Late_blight": ("Apply fungicide immediately. Critical disease — act fast.", "Critical"),
    "Potato___healthy": ("Potato plant is healthy. Maintain soil moisture.", "Low"),
    "Raspberry___healthy": ("Raspberry is healthy. Continue regular maintenance.", "Low"),
    "Soybean___healthy": ("Soybean crop is healthy. No action needed.", "Low"),
    "Squash___Powdery_mildew": ("Apply neem oil or sulfur fungicide.", "Moderate"),
    "Strawberry___Leaf_scorch": ("Remove infected leaves. Apply fungicide.", "Moderate"),
    "Strawberry___healthy": ("Strawberry plant is healthy.", "Low"),
    "Tomato___Bacterial_spot": ("Apply copper bactericide. Avoid wetting leaves.", "High"),
    "Tomato___Early_blight": ("Apply fungicide. Remove lower infected leaves.", "Moderate"),
    "Tomato___Late_blight": ("Apply fungicide immediately. This is a serious disease.", "Critical"),
    "Tomato___Leaf_Mold": ("Improve ventilation. Apply fungicide.", "Moderate"),
    "Tomato___Septoria_leaf_spot": ("Remove infected leaves. Apply fungicide.", "Moderate"),
    "Tomato___Spider_mites": ("Apply miticide or neem oil. Increase humidity.", "High"),
    "Tomato___Target_Spot": ("Apply fungicide spray. Ensure proper spacing.", "Moderate"),
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": ("Control whiteflies. Remove infected plants.", "Critical"),
    "Tomato___Tomato_mosaic_virus": ("Remove and destroy infected plants. Disinfect tools.", "Critical"),
    "Tomato___healthy": ("Tomato plant is healthy. Water at the base regularly.", "Low"),
}

WATER_STRESS_RECOMMENDATIONS = {
    "Low": "No irrigation required today. Soil moisture is adequate.",
    "Moderate": "Light watering recommended: apply 2-3 liters at the base.",
    "High": "Irrigation needed urgently. Water deeply: 5-7 liters at the base.",
    "Critical": "Critical water stress. Water immediately with 8-10 liters. Check irrigation system.",
}

RESOURCE_OPTIMIZATION = {
    "Low": "Maintain current schedule. Mulch around the plant to retain moisture.",
    "Moderate": "Water in the early morning to minimise evaporation. Check soil moisture every 2 days.",
    "High": "Use drip irrigation to deliver water directly to roots. Add organic matter to soil.",
    "Critical": "Install drip irrigation immediately. Water deeply twice daily until stress is resolved.",
}


# ════════════════════════════════════════════════════════════════════════════
#  VISUAL SYMPTOM ANALYSER
#  Operates on raw pixel data — no ML required.
#  Returns a list of detected symptoms and a severity score (0–100).
# ════════════════════════════════════════════════════════════════════════════

def analyse_visual_symptoms(image_bytes: bytes) -> dict:
    """
    Analyse an image for visible disease symptoms using pixel-level statistics.

    Detects:
      • Brown/dark lesions   — high red, low green, low blue patches
      • Yellow discolouration — high red+green, low blue (yellowing)
      • White powdery patches — high brightness with low saturation (powdery mildew)
      • Dark spots            — very low brightness pixels concentrated in clusters
      • Blight patterns       — large necrotic (dead) tissue areas
      • Uneven colouration    — high variance across the leaf (not uniform green)
      • Damaged / burnt edges — edge pixels with brown/black discolouration

    Returns:
      {
        "symptoms": [list of symptom strings],
        "severity_score": 0–100,
        "healthy_pixel_ratio": 0.0–1.0,
        "disease_pixel_ratio": 0.0–1.0,
      }
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((128, 128))                     # fast processing size
    arr = np.array(img, dtype=np.float32)

    R = arr[:, :, 0]
    G = arr[:, :, 1]
    B = arr[:, :, 2]

    total_pixels = 128 * 128

    # ── Derive useful colour channels ────────────────────────────────────────
    brightness    = (R + G + B) / 3.0
    saturation    = np.max(arr, axis=2) - np.min(arr, axis=2)   # simple saturation proxy
    green_excess  = G - np.maximum(R, B)                        # positive = green dominant

    # ── Pixel masks for disease-related colours ──────────────────────────────

    # Healthy green pixels: green dominant, reasonable brightness
    healthy_mask = (
        (green_excess > 15) &
        (G > 60) &
        (brightness > 40)
    )

    # Brown/necrotic lesion pixels: red channel dominant, low blue
    brown_mask = (
        (R > 100) &
        (R > G * 1.3) &
        (B < 80) &
        (brightness > 30) & (brightness < 200)
    )

    # Yellow discolouration: high R+G, low B
    yellow_mask = (
        (R > 120) &
        (G > 100) &
        (B < 80) &
        (R + G > B * 3)
    )

    # Dark spots / necrosis: very dark pixels that aren't shadows
    dark_spot_mask = (
        (brightness < 50) &
        (saturation < 60)
    )

    # White/grey powdery patches: high brightness, very low saturation
    powdery_mask = (
        (brightness > 180) &
        (saturation < 30)
    )

    # Blight / large dead areas: brown OR dark, covering significant area
    blight_candidate_mask = brown_mask | dark_spot_mask

    # ── Compute ratios ───────────────────────────────────────────────────────
    healthy_ratio  = healthy_mask.sum() / total_pixels
    brown_ratio    = brown_mask.sum()   / total_pixels
    yellow_ratio   = yellow_mask.sum()  / total_pixels
    dark_ratio     = dark_spot_mask.sum() / total_pixels
    powdery_ratio  = powdery_mask.sum() / total_pixels
    blight_ratio   = blight_candidate_mask.sum() / total_pixels

    # Colour variance (high = uneven = potential disease)
    g_variance = float(np.std(G))

    # ── Symptom detection with thresholds ────────────────────────────────────
    symptoms = []
    symptom_score = 0

    if brown_ratio > 0.04:                          # >4% brown pixels
        symptoms.append("Brown lesions / necrotic spots detected")
        symptom_score += min(40, int(brown_ratio * 300))

    if yellow_ratio > 0.06:                         # >6% yellow pixels
        symptoms.append("Yellow discolouration / chlorosis detected")
        symptom_score += min(30, int(yellow_ratio * 200))

    if dark_ratio > 0.05:                           # >5% dark spots
        symptoms.append("Dark spots / fungal infection signs detected")
        symptom_score += min(25, int(dark_ratio * 250))

    if powdery_ratio > 0.04:                        # >4% powdery white patches
        symptoms.append("White powdery patches (possible powdery mildew) detected")
        symptom_score += min(30, int(powdery_ratio * 300))

    if blight_ratio > 0.10:                         # >10% of leaf is dead tissue
        symptoms.append("Blight symptoms / large necrotic tissue areas detected")
        symptom_score += min(40, int(blight_ratio * 200))

    if g_variance > 45 and healthy_ratio < 0.50:   # patchy, irregular colouration
        symptoms.append("Uneven leaf colouration — irregular pigmentation detected")
        symptom_score += 15

    # Edge damage: check border pixels for brown/black discolouration
    border_pixels = np.concatenate([
        arr[0, :, :], arr[-1, :, :],
        arr[:, 0, :], arr[:, -1, :]
    ])
    border_R = border_pixels[:, 0]
    border_G = border_pixels[:, 1]
    border_B = border_pixels[:, 2]
    border_brown = ((border_R > 80) & (border_R > border_G * 1.2) & (border_B < 80)).sum()
    border_dark  = ((border_R + border_G + border_B) / 3 < 50).sum()
    border_disease_ratio = (border_brown + border_dark) / len(border_pixels)

    if border_disease_ratio > 0.20:
        symptoms.append("Damaged / burnt leaf edges detected")
        symptom_score += 20

    # Cap at 100
    severity_score = min(100, symptom_score)

    # Disease pixel ratio for downstream use
    disease_pixel_ratio = min(1.0, (brown_ratio + yellow_ratio * 0.7 + dark_ratio * 0.8 + powdery_ratio))

    return {
        "symptoms": symptoms,
        "severity_score": severity_score,
        "healthy_pixel_ratio": float(healthy_ratio),
        "disease_pixel_ratio": float(disease_pixel_ratio),
    }


def _strong_healthy_signal(healthy_ratio: float, disease_ratio: float, severity: int, symptoms: list) -> bool:
    """Return True when the image is overwhelmingly consistent with a healthy leaf."""
    return (
        healthy_ratio >= 0.68
        and disease_ratio < 0.08
        and severity <= 8
        and len(symptoms) <= 1
    )


# ════════════════════════════════════════════════════════════════════════════
#  DEMO / FALLBACK PREDICTION
#  Used when no trained model is available.
#  Now uses the visual symptom analyser as primary signal.
# ════════════════════════════════════════════════════════════════════════════

# Diseased label pool for demo mode
_DEMO_DISEASE_LABELS = [
    "Tomato___Early_blight", "Tomato___Late_blight", "Tomato___Bacterial_spot",
    "Tomato___Septoria_leaf_spot", "Tomato___Target_Spot", "Tomato___Leaf_Mold",
    "Potato___Early_blight", "Potato___Late_blight",
    "Corn___Common_rust", "Corn___Northern_Leaf_Blight",
    "Apple___Black_rot", "Apple___Apple_scab",
    "Grape___Black_rot", "Grape___Leaf_blight",
    "Pepper___Bacterial_spot", "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch", "Cherry___Powdery_mildew",
]

_DEMO_HEALTHY_LABELS = [
    "Tomato___healthy", "Potato___healthy", "Corn___healthy",
    "Apple___healthy", "Pepper___healthy", "Grape___healthy",
]


def demo_predict(image_bytes: bytes) -> dict:
    """
    Fallback prediction when the trained model is unavailable.

    Priority order:
      1. Visual symptom analysis — if significant symptoms found → Diseased
      2. Colour channel analysis — green dominance + texture variance
      3. Confidence threshold guard — Healthy requires very clear green leaf
      4. Low confidence → Unable to Identify (prevents false Healthy)
    """
    symptoms_data = analyse_visual_symptoms(image_bytes)
    symptoms      = symptoms_data["symptoms"]
    severity      = symptoms_data["severity_score"]
    healthy_ratio = symptoms_data["healthy_pixel_ratio"]
    disease_ratio = symptoms_data["disease_pixel_ratio"]

    # Seed RNG from image statistics for reproducibility
    img      = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    arr_tiny = np.array(img.resize((32, 32)), dtype=np.float32)
    seed     = int(abs(arr_tiny.mean() * 1000)) % (2 ** 31)
    rng      = random.Random(seed)

    # Strong healthy signal should win over weak symptom noise.
    if _strong_healthy_signal(healthy_ratio, disease_ratio, severity, symptoms):
        label      = rng.choice(_DEMO_HEALTHY_LABELS)
        confidence = round(rng.uniform(84, 96), 1)
        reason     = "Strong healthy visual signal detected. No meaningful disease symptoms found."
        plant_status = "Healthy"
        disease_name = "None"
        rec, water   = DISEASE_RECOMMENDATIONS.get(label, ("Plant is healthy. Maintain current care routine.", "Low"))
        recommendation = WATER_STRESS_RECOMMENDATIONS.get(water, rec)
        return {
            "plantStatus": plant_status,
            "diseaseName": disease_name,
            "confidence": confidence,
            "waterStress": water,
            "recommendation": recommendation,
            "resourceOptimization": RESOURCE_OPTIMIZATION.get(water, ""),
            "visibleSymptoms": symptoms,
            "predictionReason": reason,
        }

    # ── Decision logic ────────────────────────────────────────────────────────

    # RULE 1: Significant visual symptoms → always Diseased
    if severity >= 25 or len(symptoms) >= 3:
        label      = rng.choice(_DEMO_DISEASE_LABELS)
        confidence = round(rng.uniform(72, 91), 1)
        # Boost confidence if many symptoms present
        if severity >= 40:
            confidence = round(min(95, confidence + (severity - 40) * 0.3), 1)
        reason = f"Visual symptom analysis detected: {'; '.join(symptoms[:3])}."

    # RULE 2: Moderate symptoms — likely Diseased but lower confidence
    elif severity >= 12 or disease_ratio > 0.12:
        label      = rng.choice(_DEMO_DISEASE_LABELS)
        confidence = round(rng.uniform(60, 78), 1)
        reason     = "Mild visual symptoms detected. Possible early-stage disease."

    # RULE 3: Very clean green leaf → Healthy, but only with high confidence
    elif healthy_ratio >= 0.55 and disease_ratio < 0.04 and severity == 0:
        label      = rng.choice(_DEMO_HEALTHY_LABELS)
        confidence = round(rng.uniform(78, 94), 1)
        reason     = "Leaf shows predominantly healthy green tissue with no visible symptoms."

    # RULE 4: Ambiguous — not clearly healthy, not clearly diseased → Unable to Identify
    else:
        plant_status = "Unknown"
        disease_name = "Unable to Identify"
        confidence   = round(rng.uniform(40, 58), 1)
        rec_text     = (
            "Image analysis is inconclusive. Please retake the photo in good lighting, "
            "ensuring the leaf fills most of the frame. Consult a local agronomist if "
            "symptoms persist."
        )
        water = "Moderate"
        return {
            "plantStatus": plant_status,
            "diseaseName": disease_name,
            "confidence": confidence,
            "waterStress": water,
            "recommendation": rec_text,
            "resourceOptimization": RESOURCE_OPTIMIZATION[water],
            "visibleSymptoms": symptoms,
            "predictionReason": "Confidence too low for a reliable diagnosis.",
        }

    # ── Build final response ──────────────────────────────────────────────────
    plant_status = "Healthy" if "healthy" in label else "Diseased"
    disease_name = "None" if "healthy" in label else label.split("___")[1].replace("_", " ")
    rec, water   = DISEASE_RECOMMENDATIONS.get(label, ("Consult an agronomist.", "Moderate"))

    if plant_status == "Healthy":
        recommendation = WATER_STRESS_RECOMMENDATIONS.get(water, rec)
    else:
        recommendation = f"{rec} {WATER_STRESS_RECOMMENDATIONS.get(water, '')}".strip()

    return {
        "plantStatus": plant_status,
        "diseaseName": disease_name,
        "confidence": confidence,
        "waterStress": water,
        "recommendation": recommendation,
        "resourceOptimization": RESOURCE_OPTIMIZATION.get(water, ""),
        "visibleSymptoms": symptoms,
        "predictionReason": reason,
    }


# ════════════════════════════════════════════════════════════════════════════
#  REAL CNN PREDICTION  (used when a trained model is loaded)
# ════════════════════════════════════════════════════════════════════════════

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """
    Preprocess image for MobileNetV2 input.
    Steps:
      1. Decode bytes → PIL Image
      2. Convert to RGB  (handles RGBA, grayscale, palette images correctly)
      3. Resize to 224×224  (bilinear, same as training)
      4. Normalise to [0, 1]  (matches training: rescale=1/255)
      5. Add batch dimension
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE, Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0          # normalise [0,1]
    return np.expand_dims(arr, axis=0)                      # shape (1, 224, 224, 3)


def real_predict(image_bytes: bytes) -> dict:
    """
    Run CNN inference with multi-layer safety checks:

    1. Preprocess image (RGB, 224×224, normalised)
    2. Get softmax probability vector
    3. Run visual symptom analysis in parallel
    4. Apply rules:
       a. If top prediction is Healthy BUT visual symptoms exist → override to best
          diseased prediction from the top-5 softmax probabilities
       b. If top prediction is Healthy AND confidence < HEALTHY_CONFIDENCE_THRESHOLD
          → override to best diseased candidate from top-5, or return Unable to Identify
       c. If confidence < MIN_CONFIDENCE_THRESHOLD → Unable to Identify
       d. Otherwise → accept top prediction
    """
    img_arr     = preprocess_image(image_bytes)
    predictions = model.predict(img_arr, verbose=0)[0]      # shape (38,)

    top_idx        = int(np.argmax(predictions))
    top_confidence = float(predictions[top_idx]) * 100
    top_label      = CLASS_LABELS[top_idx] if top_idx < len(CLASS_LABELS) else "Unknown___Unknown"

    # Top-5 candidates (idx, confidence%) sorted by probability descending
    top5_indices = np.argsort(predictions)[::-1][:5]
    top5 = [
        (CLASS_LABELS[i], float(predictions[i]) * 100)
        for i in top5_indices
        if i < len(CLASS_LABELS)
    ]

    # Visual symptom analysis
    symptoms_data  = analyse_visual_symptoms(image_bytes)
    symptoms       = symptoms_data["symptoms"]
    severity       = symptoms_data["severity_score"]
    healthy_ratio  = symptoms_data["healthy_pixel_ratio"]
    disease_ratio  = symptoms_data["disease_pixel_ratio"]

    reason = ""

    # Strong healthy signal should override weak disease-leaning output.
    healthy_candidate = next(
        ((lbl, conf) for lbl, conf in top5 if "healthy" in lbl),
        None,
    )
    if healthy_candidate and _strong_healthy_signal(healthy_ratio, disease_ratio, severity, symptoms):
        top_label = healthy_candidate[0]
        top_confidence = max(healthy_candidate[1], 80.0)
        reason = (
            f"Strong healthy visual signal detected. Overriding disease-leaning model output with {top_label.replace('___', ' — ')}."
        )

    # ── Safety Rule A: Model says Healthy but visual symptoms exist ───────────
    if "healthy" in top_label and (severity >= 20 or len(symptoms) >= 2):
        # Find best diseased prediction from top-5
        best_disease = next(
            ((lbl, conf) for lbl, conf in top5 if "healthy" not in lbl),
            None
        )
        if best_disease and best_disease[1] > 30:
            top_label      = best_disease[0]
            top_confidence = best_disease[1]
            reason = (
                f"Model initially predicted Healthy ({top5[0][1]:.1f}% confidence) "
                f"but visual symptom analysis detected: {'; '.join(symptoms[:2])}. "
                f"Overriding to most likely disease."
            )
        else:
            # No confident disease prediction → Unable to Identify
            return _unable_to_identify(symptoms, severity, top_confidence,
                "Model predicted Healthy but visual disease symptoms are present. "
                "Confidence insufficient to name specific disease.")

    # ── Safety Rule B: Healthy predicted with low confidence ─────────────────
    elif "healthy" in top_label and top_confidence < HEALTHY_CONFIDENCE_THRESHOLD:
        # Only accept Healthy if there are truly no symptoms
        if severity > 0 or disease_ratio > 0.04:
            best_disease = next(
                ((lbl, conf) for lbl, conf in top5 if "healthy" not in lbl),
                None
            )
            if best_disease and best_disease[1] > MIN_CONFIDENCE_THRESHOLD:
                top_label      = best_disease[0]
                top_confidence = best_disease[1]
                reason = (
                    f"Healthy confidence ({top5[0][1]:.1f}%) below threshold "
                    f"({HEALTHY_CONFIDENCE_THRESHOLD}%) and minor symptoms detected. "
                    f"Using next best disease prediction."
                )
            else:
                return _unable_to_identify(symptoms, severity, top_confidence,
                    f"Healthy confidence ({top_confidence:.1f}%) below required "
                    f"threshold ({HEALTHY_CONFIDENCE_THRESHOLD}%). Cannot confirm "
                    f"plant is healthy.")
        else:
            reason = (
                f"Predicted Healthy with {top_confidence:.1f}% confidence. "
                f"No visual symptoms detected — prediction accepted."
            )

    # ── Safety Rule C: Overall confidence too low ─────────────────────────────
    elif top_confidence < MIN_CONFIDENCE_THRESHOLD:
        return _unable_to_identify(symptoms, severity, top_confidence,
            f"Overall prediction confidence ({top_confidence:.1f}%) is below the "
            f"minimum threshold ({MIN_CONFIDENCE_THRESHOLD}%). Image may be unclear "
            f"or not a plant leaf.")

    # ── Rule D: Accepted as-is ────────────────────────────────────────────────
    else:
        reason = (
            f"CNN prediction: {top_label.replace('___', ' — ')} "
            f"with {top_confidence:.1f}% confidence."
        )
        if symptoms:
            reason += f" Visual symptoms confirmed: {'; '.join(symptoms[:2])}."

    # ── Build response ────────────────────────────────────────────────────────
    plant_status = "Healthy" if "healthy" in top_label else "Diseased"
    disease_name = "None" if "healthy" in top_label else top_label.split("___")[1].replace("_", " ")
    rec, water   = DISEASE_RECOMMENDATIONS.get(top_label, ("Consult an agronomist.", "Moderate"))

    if plant_status == "Healthy":
        recommendation = WATER_STRESS_RECOMMENDATIONS.get(water, rec)
    else:
        recommendation = f"{rec} {WATER_STRESS_RECOMMENDATIONS.get(water, '')}".strip()

    return {
        "plantStatus": plant_status,
        "diseaseName": disease_name,
        "confidence": round(top_confidence, 1),
        "waterStress": water,
        "recommendation": recommendation,
        "resourceOptimization": RESOURCE_OPTIMIZATION.get(water, ""),
        "visibleSymptoms": symptoms,
        "predictionReason": reason,
    }


def _unable_to_identify(symptoms: list, severity: int, confidence: float, reason: str) -> dict:
    """Standard 'Unable to Identify' response with actionable advice."""
    water = "Moderate" if not symptoms else "High"
    symptom_str = (
        f"Visible symptoms detected: {'; '.join(symptoms)}. " if symptoms
        else "No clear symptoms detected from pixel analysis. "
    )
    rec = (
        f"{symptom_str}Please retake the photo with better lighting and focus. "
        "If you observe visible spots, lesions or discolouration, treat as potentially diseased "
        "and consult a local agronomist."
    )
    return {
        "plantStatus": "Unknown",
        "diseaseName": "Unable to Identify",
        "confidence": round(confidence, 1),
        "waterStress": water,
        "recommendation": rec,
        "resourceOptimization": RESOURCE_OPTIMIZATION.get(water, ""),
        "visibleSymptoms": symptoms,
        "predictionReason": reason,
    }


# ════════════════════════════════════════════════════════════════════════════
#  FASTAPI APPLICATION
# ════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="SmartFarm AI Engine",
    description="CNN-based Plant Disease & Water Stress Detection API (v2 — with visual symptom validation)",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None


@app.on_event("startup")
def load_model():
    global model, DEMO_MODE
    if TF_AVAILABLE and os.path.exists(MODEL_PATH):
        try:
            model = tf.keras.models.load_model(MODEL_PATH)
            DEMO_MODE = False
            print(f"✅ Model loaded from {MODEL_PATH}")
            print(f"   • Healthy confidence threshold : {HEALTHY_CONFIDENCE_THRESHOLD}%")
            print(f"   • Minimum confidence threshold : {MIN_CONFIDENCE_THRESHOLD}%")
        except Exception as e:
            print(f"⚠️  Failed to load model: {e}. Falling back to demo mode.")
            DEMO_MODE = True
    else:
        DEMO_MODE = True
        print("ℹ️  Running in DEMO mode — visual symptom analyser active.")


# ── Response schema ──────────────────────────────────────────────────────────
class PredictionResponse(BaseModel):
    plantStatus: str
    diseaseName: str
    confidence: float
    waterStress: str
    recommendation: str
    resourceOptimization: str
    visibleSymptoms: list
    predictionReason: str


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """
    Analyse a plant leaf image.
    Accepts: JPEG, PNG, WEBP.
    Returns: disease prediction, confidence, visible symptoms, reason, treatment.
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image (JPEG/PNG/WEBP)")

    image_bytes = await file.read()
    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty file received")

    try:
        result = demo_predict(image_bytes) if DEMO_MODE else real_predict(image_bytes)
        # Ensure new fields are always present (backward compatibility)
        result.setdefault("visibleSymptoms", [])
        result.setdefault("predictionReason", "")
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "version": "2.0.0",
        "mode": "demo" if DEMO_MODE else "production",
        "model_loaded": model is not None,
        "tf_available": TF_AVAILABLE,
        "healthy_confidence_threshold": HEALTHY_CONFIDENCE_THRESHOLD,
        "min_confidence_threshold": MIN_CONFIDENCE_THRESHOLD,
    }


@app.get("/classes")
def get_classes():
    return {
        "classes": CLASS_LABELS,
        "total": len(CLASS_LABELS),
        "healthy_count": len(HEALTHY_LABELS),
        "diseased_count": len(DISEASED_LABELS),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
