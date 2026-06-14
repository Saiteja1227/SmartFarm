"""
History Blueprint  /api/history  — public, no authentication required.
"""

from flask import Blueprint, request, jsonify, current_app
from bson import ObjectId
from bson.errors import InvalidId

from app.extensions import mongo
from app.models.analysis import serialize
from app.storage import delete_local_analysis, get_local_analysis, list_local_analyses

history_bp = Blueprint("history", __name__)


@history_bp.route("/", methods=["GET"])
def get_history():
    try:
        page  = max(int(request.args.get("page",  1)),  1)
        limit = max(int(request.args.get("limit", 20)), 1)
        skip  = (page - 1) * limit

        if current_app.config.get("MONGO_AVAILABLE", False):
            total = mongo.db.analyses.count_documents({})
            cursor = (
                mongo.db.analyses.find()
                .sort("created_at", -1)
                .skip(skip)
                .limit(limit)
            )
            records = [serialize(doc) for doc in cursor]
        else:
            local_records = sorted(
                list_local_analyses(),
                key=lambda record: str(record.get("created_at", "")),
                reverse=True,
            )
            total = len(local_records)
            records = [serialize(doc) for doc in local_records[skip:skip + limit]]

        return jsonify({
            "success": True,
            "data": records,
            "pagination": {
                "page":  page,
                "limit": limit,
                "total": total,
                "pages": max(1, -(-total // limit)),
            },
        })
    except Exception as exc:
        current_app.logger.error("History fetch error: %s", exc)
        return jsonify({"success": False, "message": "Failed to fetch history"}), 500


@history_bp.route("/<record_id>", methods=["GET"])
def get_record(record_id):
    try:
        doc = None
        if current_app.config.get("MONGO_AVAILABLE", False):
            doc = mongo.db.analyses.find_one({"_id": ObjectId(record_id)})
        else:
            doc = None

        if doc is None and current_app.config.get("MONGO_AVAILABLE", False):
            try:
                doc = mongo.db.analyses.find_one({"_id": ObjectId(record_id)})
            except InvalidId:
                doc = None
            except Exception as db_exc:
                current_app.logger.warning("Record fetch fallback to local store: %s", db_exc)

        if doc is None:
            doc = get_local_analysis(record_id)

        if doc is None:
            return jsonify({"success": False, "message": "Record not found"}), 404

        return jsonify({"success": True, "data": serialize(doc)})
    except Exception as exc:
        current_app.logger.error("Record fetch error: %s", exc)
        return jsonify({"success": False, "message": "Failed to fetch record"}), 500


@history_bp.route("/<record_id>", methods=["DELETE"])
def delete_record(record_id):
    try:
        result = None
        if current_app.config.get("MONGO_AVAILABLE", False):
            result = mongo.db.analyses.find_one_and_delete({"_id": ObjectId(record_id)})
        else:
            result = None

        if result is None and current_app.config.get("MONGO_AVAILABLE", False):
            try:
                result = mongo.db.analyses.find_one_and_delete({"_id": ObjectId(record_id)})
            except InvalidId:
                result = None
            except Exception as db_exc:
                current_app.logger.warning("Delete fallback to local store: %s", db_exc)

        if result is None:
            result = delete_local_analysis(record_id)

        if result is None:
            return jsonify({"success": False, "message": "Record not found"}), 404

        return jsonify({"success": True, "message": "Record deleted successfully"})
    except Exception as exc:
        current_app.logger.error("Delete error: %s", exc)
        return jsonify({"success": False, "message": "Failed to delete record"}), 500
