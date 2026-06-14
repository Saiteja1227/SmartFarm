# 🌿 SmartFarm AI

**AI-Driven Smart Urban Farming Resource Optimisation**

SmartFarm AI is a web application that analyses plant leaf images using a CNN-based deep learning model to detect diseases, assess water stress levels, and provide actionable recommendations — all without any sensors or manual input.

---

## Features

- 📷 **Drag & drop image upload** — JPEG, PNG, WEBP supported
- 🤖 **AI disease detection** — MobileNetV2 CNN trained on 54,000+ PlantVillage images
- 💧 **Water stress analysis** — Low / Moderate / High / Critical levels
- 🔍 **Visual symptom detection** — spots, lesions, blight, yellowing, powdery mildew
- 📊 **Admin dashboard** — charts for disease trends, water stress distribution, activity
- 🕐 **Analysis history** — paginated records with filter and delete
- 🔊 **Voice output** — Web Speech API reads results aloud
- ☁️ **MongoDB Atlas** — cloud database, production-ready

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Flask + Jinja2, Vanilla JS, Recharts (CDN) |
| Backend | Python 3.14, Flask 3.0 |
| ML Server | FastAPI + MobileNetV2 (TensorFlow) |
| Database | MongoDB Atlas (Free Tier) |
| Deployment | Gunicorn |

---

## Project Structure

```
python-app/
├── run.py                    # Flask entry point
├── .env.example              # Environment variable template
├── requirements.txt
├── app/
│   ├── __init__.py           # App factory
│   ├── extensions.py         # PyMongo instance
│   ├── models/analysis.py    # MongoDB document helpers
│   ├── routes/
│   │   ├── main.py           # Page routes
│   │   ├── analyze.py        # POST /api/analyze
│   │   ├── history.py        # GET/DELETE /api/history
│   │   └── stats.py          # GET /api/stats
│   ├── templates/            # Jinja2 HTML pages
│   └── static/               # CSS, JS, uploaded images
└── ml_model/
    ├── server.py             # FastAPI inference server
    └── model.py              # CNN training script
```

---

## Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/Saiteja1227/SmartFarm.git
cd SmartFarm/python-app
```

### 2. Create virtual environment
```bash
python3 -m venv venv
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate         # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure environment
```bash
cp .env.example .env
# Edit .env and add your MongoDB Atlas URI
```

### 5. Run the app
```bash
python run.py
```
Open **http://localhost:5001**

### 6. (Optional) Start ML inference server
```bash
cd ml_model
python server.py
```
The app works without it using the built-in visual symptom analyser fallback.

---

## API Endpoints

| Method | URL | Description |
|---|---|---|
| `POST` | `/api/analyze/` | Upload image → AI prediction |
| `GET` | `/api/history/` | Paginated analysis history |
| `GET` | `/api/history/<id>` | Single analysis record |
| `DELETE` | `/api/history/<id>` | Delete a record |
| `GET` | `/api/stats/` | Dashboard statistics |
| `GET` | `/api/health` | Health check |

---

## License

MIT
