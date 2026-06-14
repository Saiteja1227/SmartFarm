# SmartFarm AI — Python/Flask Port

> AI-Driven Smart Urban Farming Resource Optimization  
> Complete migration from React/Express.js → Flask/Jinja2/Python

---

## Project Structure

```
python-app/
├── run.py                        # Flask entry point  (replaces backend/src/index.js)
├── .env                          # Environment variables
├── requirements.txt              # All Python dependencies
│
├── app/
│   ├── __init__.py               # Flask app factory + MongoDB init
│   ├── models/
│   │   └── analysis.py           # Document builder + serializer  (replaces Analysis.js Mongoose model)
│   ├── routes/
│   │   ├── main.py               # HTML page routes               (replaces React Router)
│   │   ├── analyze.py            # POST /api/analyze              (replaces routes/analyze.js)
│   │   ├── history.py            # GET/DELETE /api/history        (replaces routes/history.js)
│   │   └── stats.py              # GET /api/stats                 (replaces routes/stats.js)
│   ├── templates/
│   │   ├── base.html             # Shared layout + Navbar + Footer (replaces App.js + Navbar.js)
│   │   ├── home.html             # Home page                      (replaces HomePage.js)
│   │   ├── upload.html           # Upload & analyse               (replaces UploadPage.js)
│   │   ├── results.html          # Analysis results + voice       (replaces ResultsPage.js)
│   │   ├── history.html          # Paginated history              (replaces HistoryPage.js)
│   │   └── admin.html            # Dashboard + Recharts           (replaces AdminDashboard.js)
│   └── static/
│       ├── css/main.css          # Full stylesheet                (replaces index.css + CSS-in-JS)
│       ├── js/main.js            # Navbar + toast helpers         (replaces react-hot-toast + Navbar state)
│       └── uploads/              # Uploaded images (served statically)
│
└── ml_model/
    ├── model.py                  # CNN training script            (unchanged)
    ├── server.py                 # FastAPI inference server       (unchanged)
    └── requirements.txt          # ML-specific dependencies
```

---

## React → Python Mapping

| React Component / Node.js File | Python Equivalent |
|---|---|
| `backend/src/index.js` (Express server) | `run.py` + `app/__init__.py` (Flask app factory) |
| `backend/src/models/Analysis.js` (Mongoose) | `app/models/analysis.py` (PyMongo document helpers) |
| `backend/src/routes/analyze.js` | `app/routes/analyze.py` |
| `backend/src/routes/history.js` | `app/routes/history.py` |
| `backend/src/routes/stats.js` | `app/routes/stats.py` |
| `frontend/src/App.js` + React Router | `app/routes/main.py` (Flask routes) + `base.html` |
| `frontend/src/components/Navbar.js` | `base.html` navbar + `main.js` hamburger logic |
| `frontend/src/pages/HomePage.js` | `templates/home.html` |
| `frontend/src/pages/UploadPage.js` | `templates/upload.html` + inline `<script>` |
| `frontend/src/pages/ResultsPage.js` | `templates/results.html` + inline `<script>` |
| `frontend/src/pages/HistoryPage.js` | `templates/history.html` + inline `<script>` |
| `frontend/src/pages/AdminDashboard.js` | `templates/admin.html` + Recharts CDN |
| `frontend/src/api/index.js` (axios) | Native `fetch()` calls in each template's `<script>` |
| `react-hot-toast` | `showToast()` in `static/js/main.js` |
| CSS-in-JS (inline styles) | `static/css/main.css` |
| `ml-model/model.py` | `ml_model/model.py` (unchanged) |
| `ml-model/server.py` (FastAPI) | `ml_model/server.py` (unchanged) |

---

## Prerequisites

- Python 3.10+
- MongoDB running locally (`mongodb://localhost:27017`)
- pip

---

## Installation & Setup

### 1. Clone / navigate to the python-app folder
```bash
cd /path/to/MP/python-app
```

### 2. Create and activate a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate         # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure environment variables
Edit `.env` (already pre-filled with sensible defaults):
```
FLASK_ENV=development
FLASK_DEBUG=1
SECRET_KEY=your-secret-key-here
MONGODB_URI=mongodb://localhost:27017/smartfarming
ML_MODEL_URL=http://localhost:8000
PORT=5001
```

### 5. Start MongoDB
```bash
mongod --dbpath /usr/local/var/mongodb   # macOS Homebrew
# or: brew services start mongodb-community
```

---

## Running the Application

### Flask web app (port 5001)
```bash
python run.py
```
Open http://localhost:5001

### ML inference server (port 8000) — optional, demo mode works without it
```bash
cd ml_model
pip install -r requirements.txt
python server.py
```

### Production (Gunicorn)
```bash
gunicorn -w 4 -b 0.0.0.0:5001 "run:app"
```

---

## API Endpoints (unchanged from Node.js version)

| Method | URL | Description |
|--------|-----|-------------|
| `POST` | `/api/analyze/` | Upload image → ML prediction → MongoDB save |
| `GET`  | `/api/history/` | Paginated analysis history |
| `GET`  | `/api/history/<id>` | Single analysis record |
| `DELETE` | `/api/history/<id>` | Delete a record |
| `GET`  | `/api/stats/` | Aggregated dashboard statistics |
| `GET`  | `/api/health` | Health check |

---

## Features Preserved

- ✅ Drag-and-drop image upload with live preview
- ✅ AI plant disease detection (CNN via FastAPI ML server)
- ✅ Water stress level assessment
- ✅ Smart resource optimization recommendations
- ✅ Voice output (Web Speech API — browser native, no change needed)
- ✅ Analysis history with pagination, filter, and delete
- ✅ Admin dashboard with Pie, Bar, and Line charts (Recharts via CDN)
- ✅ Responsive navbar with mobile hamburger menu
- ✅ Toast notifications
- ✅ MongoDB persistence (same database, same collection schema)
- ✅ Mock/demo fallback when ML server is offline
- ✅ Health check endpoint

---

## Features That Changed and Why

| React Feature | Python Alternative | Reason |
|---|---|---|
| React state (`useState`) | Vanilla JS variables in `<script>` blocks | No React runtime needed; logic is identical |
| React Router (`<Routes>`) | Flask `@app.route()` decorators | Server-side routing is the Flask way |
| axios HTTP client | Native `fetch()` API | Built into all modern browsers; no extra dependency |
| CSS-in-JS (inline `style={{}}`) | `static/css/main.css` class names | Better separation of concerns |
| `react-hot-toast` | Custom `showToast()` in `main.js` | 20-line replacement with identical UX |
| Mongoose ODM | PyMongo + helper functions | Direct MongoDB driver; Mongoose-equivalent schema validation in Python |
| `recharts` npm package | Recharts CDN + React CDN | Same charts, zero build step |
| `react-dropzone` | Native drag-and-drop events | Full feature parity with ~40 lines of vanilla JS |
| `lucide-react` icons | Emoji equivalents in HTML | No SVG library needed for server-rendered HTML |

---

## Notes on Production Deployment

1. Set `FLASK_DEBUG=0` and `FLASK_ENV=production` in `.env`
2. Use a strong random `SECRET_KEY`
3. Run behind Nginx + Gunicorn (see command above)
4. Serve `app/static/` via Nginx for better performance
5. Use MongoDB Atlas for managed cloud database
