"""
Stats Blueprint
GET /api/stats — aggregated statistics for the Admin Dashboard

Mirrors: backend/src/routes/stats.js
All MongoDB aggregation pipelines are ported 1-to-1 using PyMongo.
"""

from datetime import datetime, timezone, timedelta
from flask import Blueprint, jsonify, current_app

from app.extensions import mongo
from app.storage import list_local_analyses

stats_bp = Blueprint("stats", __name__)


@stats_bp.route("/", methods=["GET"])
def get_stats():
    try:
        if current_app.config.get("MONGO_AVAILABLE", False):
            db = mongo.db.analyses

            # ── Basic counts ───────────────────────────────────────────────
            total_uploads = db.count_documents({})
            healthy_count = db.count_documents({"plant_status": "Healthy"})
            diseased_count = db.count_documents({"plant_status": "Diseased"})

            # ── Top 5 diseases ────────────────────────────────────────────
            disease_pipeline = [
                {"$match": {"plant_status": "Diseased"}},
                {"$group": {"_id": "$disease_name", "count": {"$sum": 1}}},
                {"$sort": {"count": -1}},
                {"$limit": 5},
                {"$project": {"name": "$_id", "count": 1, "_id": 0}},
            ]
            top_diseases = list(db.aggregate(disease_pipeline))

            # ── Water stress distribution ─────────────────────────────────
            water_pipeline = [
                {"$group": {"_id": "$water_stress_level", "count": {"$sum": 1}}},
                {"$project": {"name": "$_id", "count": 1, "_id": 0}},
            ]
            water_distribution = list(db.aggregate(water_pipeline))

            # ── Last 7 days activity ──────────────────────────────────────
            seven_days_ago = datetime.now(timezone.utc) - timedelta(days=7)
            activity_pipeline = [
                {"$match": {"created_at": {"$gte": seven_days_ago}}},
                {
                    "$group": {
                        "_id": {
                            "$dateToString": {"format": "%Y-%m-%d", "date": "$created_at"}
                        },
                        "count": {"$sum": 1},
                    }
                },
                {"$sort": {"_id": 1}},
                {"$project": {"date": "$_id", "count": 1, "_id": 0}},
            ]
            recent_activity = list(db.aggregate(activity_pipeline))

            # ── Average confidence ────────────────────────────────────────
            avg_pipeline = [{"$group": {"_id": None, "avg": {"$avg": "$confidence"}}}]
            avg_result = list(db.aggregate(avg_pipeline))
            avg_confidence = round(avg_result[0]["avg"], 1) if avg_result else 0.0
        else:
            records = list_local_analyses()
            total_uploads = len(records)
            healthy_count = sum(1 for record in records if record.get("plant_status") == "Healthy")
            diseased_count = sum(1 for record in records if record.get("plant_status") == "Diseased")

            disease_counts = {}
            water_counts = {}
            recent_counts = {}
            seven_days_ago = datetime.now(timezone.utc) - timedelta(days=7)

            for record in records:
                if record.get("plant_status") == "Diseased":
                    disease_name = record.get("disease_name") or "Unknown"
                    disease_counts[disease_name] = disease_counts.get(disease_name, 0) + 1

                water_level = record.get("water_stress_level") or "Unknown"
                water_counts[water_level] = water_counts.get(water_level, 0) + 1

                created_at = record.get("created_at")
                if created_at:
                    try:
                        created_dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                    except Exception:
                        created_dt = None
                    if created_dt and created_dt >= seven_days_ago:
                        day_key = created_dt.astimezone(timezone.utc).strftime("%Y-%m-%d")
                        recent_counts[day_key] = recent_counts.get(day_key, 0) + 1

            top_diseases = [
                {"name": name, "count": count}
                for name, count in sorted(disease_counts.items(), key=lambda item: item[1], reverse=True)[:5]
            ]
            water_distribution = [
                {"name": name, "count": count}
                for name, count in sorted(water_counts.items(), key=lambda item: item[0])
            ]
            recent_activity = [
                {"date": date, "count": count}
                for date, count in sorted(recent_counts.items())
            ]
            avg_confidence = round(
                sum(float(record.get("confidence", 0)) for record in records) / total_uploads, 1
            ) if total_uploads else 0.0

        # ── Percentages ───────────────────────────────────────────────────
        healthy_pct = round((healthy_count / total_uploads) * 100, 1) if total_uploads else 0.0
        diseased_pct = round((diseased_count / total_uploads) * 100, 1) if total_uploads else 0.0

        return jsonify({
            "success": True,
            "data": {
                "totalUploads": total_uploads,
                "healthyCount": healthy_count,
                "diseasedCount": diseased_count,
                "healthyPercentage": healthy_pct,
                "diseasedPercentage": diseased_pct,
                "avgConfidence": avg_confidence,
                "topDiseases": top_diseases,
                "waterStressDistribution": water_distribution,
                "recentActivity": recent_activity,
            },
        })

    except Exception as exc:
        current_app.logger.error("Stats error: %s", exc)
        return jsonify({"success": False, "message": "Failed to fetch statistics"}), 500
