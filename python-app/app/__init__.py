"""
SmartFarm AI — Flask Application Factory
No authentication — fully open, public access to all features.
"""

import os
import certifi
from flask import Flask, jsonify
from dotenv import load_dotenv

load_dotenv()

from app.extensions import mongo


def create_app():
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.config["MONGO_AVAILABLE"] = False

    # ── Core config ────────────────────────────────────────────────────────
    app.config["SECRET_KEY"]         = os.getenv("SECRET_KEY", "smartfarm-dev-key")
    app.config["MAX_CONTENT_LENGTH"] = int(os.getenv("MAX_CONTENT_LENGTH", 10 * 1024 * 1024))
    app.config["UPLOAD_FOLDER"]      = os.path.join(os.path.dirname(__file__), "static", "uploads")
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

    # ── MongoDB Atlas ──────────────────────────────────────────────────────
    raw_uri = os.getenv("MONGODB_URI")
    if raw_uri and not raw_uri.startswith(("mongodb://localhost", "mongodb://127.0.0.1", "mongodb://[::1]", "mongodb://0.0.0.0")):
        if "mongodb+srv" in raw_uri and "tlsCAFile" not in raw_uri:
            sep = "&" if "?" in raw_uri else "?"
            raw_uri = f"{raw_uri}{sep}tlsCAFile={certifi.where()}"

        app.config["MONGO_URI"] = raw_uri
        mongo.init_app(app)
        app.config["MONGO_AVAILABLE"] = True
    else:
        app.logger.info("MONGODB_URI not set, using local fallback store.")

    # ── Register blueprints ────────────────────────────────────────────────
    from app.routes.main    import main_bp
    from app.routes.analyze import analyze_bp
    from app.routes.history import history_bp
    from app.routes.stats   import stats_bp

    app.register_blueprint(main_bp)
    app.register_blueprint(analyze_bp, url_prefix="/api/analyze")
    app.register_blueprint(history_bp, url_prefix="/api/history")
    app.register_blueprint(stats_bp,   url_prefix="/api/stats")

    # ── Security response headers ──────────────────────────────────────────
    @app.after_request
    def set_security_headers(response):
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"]         = "SAMEORIGIN"
        response.headers["X-XSS-Protection"]        = "1; mode=block"
        return response

    # ── Error handlers ─────────────────────────────────────────────────────
    @app.errorhandler(404)
    def not_found(e):
        return jsonify({"success": False, "message": "Route not found"}), 404

    @app.errorhandler(500)
    def server_error(e):
        return jsonify({"success": False, "message": "Internal server error"}), 500

    @app.errorhandler(413)
    def too_large(e):
        return jsonify({"success": False, "message": "File too large. Max 10 MB."}), 413

    return app


def _create_indexes():
    """Create MongoDB indexes for query performance."""
    try:
        db = mongo.db
        db.analyses.create_index("plant_status")
        db.analyses.create_index("created_at")
    except Exception as exc:
        import logging
        logging.warning("Index creation warning: %s", exc)
