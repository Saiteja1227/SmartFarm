"""
SmartFarm AI - Flask Application Entry Point
Run:  python run.py
Prod: gunicorn -w 4 -b 0.0.0.0:5001 "run:app"
"""

import os
from app import create_app

app = create_app()

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5001))
    debug = os.getenv("FLASK_DEBUG", "1") == "1"
    print(f"🚀 SmartFarm AI Flask server running on http://localhost:{port}")
    print(f"📡 ML Model endpoint: {os.getenv('ML_MODEL_URL', 'http://localhost:8000')}")
    app.run(host="0.0.0.0", port=port, debug=debug)
