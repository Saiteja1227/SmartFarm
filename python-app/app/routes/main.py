"""
Main Blueprint — all page routes are public, no login required.
"""

from flask import Blueprint, render_template, jsonify
from datetime import datetime, timezone

main_bp = Blueprint("main", __name__)


@main_bp.route("/")
def home():
    return render_template("home.html")


@main_bp.route("/upload")
def upload():
    return render_template("upload.html")


@main_bp.route("/results/<string:record_id>")
def results(record_id):
    return render_template("results.html", record_id=record_id)


@main_bp.route("/history")
def history():
    return render_template("history.html")


@main_bp.route("/admin")
def admin():
    return render_template("admin.html")


@main_bp.route("/api/health")
def health():
    return jsonify({"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()})
