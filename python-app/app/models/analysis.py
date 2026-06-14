"""
Analysis Model helpers
All documents stored in the 'analyses' collection.

v2 additions:
  - visible_symptoms  : list[str]  — symptoms detected by visual analyser
  - prediction_reason : str        — explanation of why this result was chosen
"""

from datetime import datetime, timezone
from bson import ObjectId

VALID_PLANT_STATUS = {"Healthy", "Diseased", "Unknown"}
VALID_WATER_STRESS = {"Low", "Moderate", "High", "Critical"}


def build_document(
    image_url: str,
    image_preview: str = "",
    image_name: str = "plant_image.jpg",
    plant_status: str = "Unknown",
    disease_name: str = "None",
    confidence: float = 0.0,
    water_stress_level: str = "Low",
    recommendation: str = "No recommendation available.",
    resource_optimization: str = "",
    visible_symptoms: list = None,
    prediction_reason: str = "",
) -> dict:
    """Create a new analysis document ready to insert into MongoDB."""
    now = datetime.now(timezone.utc)
    return {
        "image_url":           image_url,
        "image_preview":       image_preview,
        "image_name":          image_name,
        "plant_status":        plant_status if plant_status in VALID_PLANT_STATUS else "Unknown",
        "disease_name":        disease_name,
        "confidence":          round(float(confidence), 1),
        "water_stress_level":  water_stress_level if water_stress_level in VALID_WATER_STRESS else "Low",
        "recommendation":      recommendation,
        "resource_optimization": resource_optimization,
        "visible_symptoms":    visible_symptoms or [],
        "prediction_reason":   prediction_reason,
        "created_at":          now,
        "updated_at":          now,
    }


def serialize(doc: dict) -> dict:
    """Convert a MongoDB document to a JSON-serialisable dict."""
    if doc is None:
        return None
    result = dict(doc)
    if "_id" in result:
        result["_id"] = str(result["_id"])
    for field in ("created_at", "updated_at"):
        if field in result and isinstance(result[field], datetime):
            result[field] = result[field].isoformat()
    # Ensure new fields always present in serialised output
    result.setdefault("image_preview", "")
    result.setdefault("visible_symptoms", [])
    result.setdefault("prediction_reason", "")
    return result
