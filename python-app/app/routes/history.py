"""
History Blueprint  /api/history  — public, no authentication required.
"""

from flask import Blueprint, request, jsonify, current_app
from bson import ObjectId
from bson.errors import InvalidId

from app.extensions import mongo
from app.models.analysis import serialize

history_bp = Blueprint("history", __name__)


@history_bp.route("/", methods=["GET"])
def get_history():
    try:
        page  = max(int(request.args.get("page",  1)),  1)
        limit = max(int(request.args.get("limit", 20)), 1)
        skip  = (page - 1) * limit

        total  = mongo.db.analyses.count_documents({})
        cursor = (
            mongo.db.analyses.find()
            .sort("created_at", -1)
            .skip(skip)
            .limit(limit)
        )
        records = [serialize(doc) for doc in cursor]
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
        doc = mongo.db.analyses.find_one({"_id": ObjectId(record_id)})
        if doc is None:
            return jsonify({"success": False, "message": "Record not found"}), 404
        return jsonify({"success": True, "data": serialize(doc)})
    except InvalidId:
        return jsonify({"success": False, "message": "Invalid record ID"}), 400
    except Exception as exc:
        current_app.logger.error("Record fetch error: %s", exc)
        return jsonify({"success": False, "message": "Failed to fetch record"}), 500


@history_bp.route("/<record_id>", methods=["DELETE"])
def delete_record(record_id):
    try:
        result = mongo.db.analyses.find_one_and_delete({"_id": ObjectId(record_id)})
        if result is None:
            return jsonify({"success": False, "message": "Record not found"}), 404
        return jsonify({"success": True, "message": "Record deleted successfully"})
    except InvalidId:
        return jsonify({"success": False, "message": "Invalid record ID"}), 400
    except Exception as exc:
        current_app.logger.error("Delete error: %s", exc)
        return jsonify({"success": False, "message": "Failed to delete record"}), 500
