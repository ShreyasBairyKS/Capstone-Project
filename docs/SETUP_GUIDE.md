# VisionFood QAI — Full Setup Guide

> **Purpose:** Get the entire VisionFood QAI stack (backend API + MongoDB + frontend dashboard) running locally.
> This guide tells you exactly what state the codebase is in, what will and won't work right now, and the exact steps to set up everything.

For bottle-cap model training/evaluation/inference on RTX A5000 + 128GB RAM, use:
- `docs/BOTTLE_CAP_A5000_GUIDE.md`

---

## ⚠️ Current State — What Works Without Setup

| Component | Status | Notes |
|---|---|---|
| Frontend renders login page | ✅ Works | No backend needed for the login UI to render |
| Frontend dashboard UI | ⚠️ Partial | Renders tabs/navigation but all API calls will fail with network errors if backend is not running |
| Backend starts without MongoDB | ⚠️ Partial | FastAPI starts, but any `/api/v1/products` or `/api/v1/runs` call will throw `RuntimeError: Motor database is not initialised` |
| Backend inspection endpoint | ✅ Works (with SQLite) | Uses SQLite by default — no SQLite setup needed, file is auto-created |
| Frontend ↔ Backend connection | ❌ Broken until configured | Vite proxies `/api` → `localhost:8000` — backend must be running |
| Authentication (Google OAuth) | ❌ Not wired up | JWT auth is planned but not yet active; current `X-API-Key` header auth is used instead |

---

## Prerequisites — Install These First

### 1. Python 3.11.x

The project **requires Python 3.11**. 3.12+ will work but is untested.

```
Download: https://www.python.org/downloads/release/python-3119/
```

Verify:
```powershell
python --version
# Should output: Python 3.11.x
```

### 2. Node.js 20 LTS

Required to run the React/Vite frontend.

```
Download: https://nodejs.org/en/download
```

Verify:
```powershell
node --version    # v20.x.x
npm --version     # 10.x.x
```

### 3. MongoDB Community Server 7.0

The new Products and Production Runs data is stored in MongoDB. The existing inspection logs use SQLite (no extra setup needed for SQLite).

**Option A — Local install (recommended for dev):**
```
Download: https://www.mongodb.com/try/download/community
```

During install:
- Select **"Install MongoDB as a Service"** ✅
- Leave data directory as default (`C:\data\db`)
- Install **MongoDB Compass** (GUI) alongside — useful for inspecting data

Verify MongoDB is running:
```powershell
# In an admin PowerShell
Get-Service MongoDB
# Status should be: Running
```

**Option B — Docker (no Windows service install):**
```powershell
docker run -d --name visionfood-mongo -p 27017:27017 mongo:7.0
```

**Option C — MongoDB Atlas (cloud, no local install):**
- Create free cluster at https://cloud.mongodb.com
- Get the connection string: `mongodb+srv://user:password@cluster.mongodb.net/`
- You'll use this in the `.env` file as `MONGO_URL`

---

## Step 1 — Clone and Open Project

```powershell
# The project is already at:
cd "d:\Capstone Project code\Capstone-Project"
```

---

## Step 2 — Backend Environment Setup (`.env`)

The `.env.example` file at the project root already exists. Copy it and fill in values.

```powershell
# From project root
Copy-Item .env.example .env
```

Now open `.env` and update it to match the following complete configuration:

```ini
# ============================================================
# VisionFood QAI — local development .env
# ============================================================

# --- Deployment ---
TIER=edge
DEVICE_ID=edge_node_01

# --- Model Paths (leave as-is; models load lazily — missing files won't crash startup) ---
YOLOV11_ONNX_PATH=models/yolov11n_best.onnx
EFFICIENTVIT_ONNX_PATH=models/efficientvit_m5_best.onnx

# --- Inference Thresholds ---
YOLOV11_CONF_THRESHOLD=0.40
YOLOV11_IOU_THRESHOLD=0.45
CONFIRMED_DEFECT_THRESHOLD=0.85
ESCALATE_THRESHOLD=0.60
HUMAN_REVIEW_THRESHOLD=0.45

# --- Camera ---
CAMERA_INDEX=0
CAMERA_MODE=software

# -------------------------------------------------------------------
# SQLite (inspection logs — auto-created, no setup needed)
DATABASE_URL=sqlite:///./visionfood_dev.db

# -------------------------------------------------------------------
# MongoDB (products + production runs)
# If using local MongoDB:
MONGO_URL=mongodb://localhost:27017
MONGO_DB_NAME=visionfood

# If using MongoDB Atlas, replace with:
# MONGO_URL=mongodb+srv://YOUR_USER:YOUR_PASS@cluster0.xxxxx.mongodb.net/?retryWrites=true&w=majority
# MONGO_DB_NAME=visionfood

# -------------------------------------------------------------------
# Redis (live-stream WebSocket feed — dashboard Live tab)
# If Redis is NOT installed yet, set REMEDY_ENABLED=false and comment this out
REDIS_URL=redis://localhost:6379

# -------------------------------------------------------------------
# API Security
# Generate a real key with: python -c "import secrets; print(secrets.token_hex(32))"
# For local dev you can leave this as-is — it must match VITE_API_KEY in the frontend .env
API_KEY=dev-insecure-key

# -------------------------------------------------------------------
# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json

# -------------------------------------------------------------------
# REMEDY Engine
REMEDY_ENABLED=true

# -------------------------------------------------------------------
# Optional — leave blank if not using these services
WANDB_API_KEY=
```

> **Note on Redis:** The Live tab WebSocket feed requires Redis. If you haven't installed Redis yet, the backend will still start but the live stream will not work. See Step 7 for Redis setup.

---

## Step 3 — Frontend Environment Setup (`.env`)

There's a separate `.env` file inside the `dashboard/` folder.

```powershell
# From project root
Copy-Item dashboard\.env.example dashboard\.env
```

Open `dashboard/.env` and make sure it contains:

```ini
# Vite dev server proxies /api → localhost:8000;
# leave VITE_API_BASE as /api for local dev
VITE_API_BASE=/api

# Must exactly match the API_KEY value in the root .env file
VITE_API_KEY=dev-insecure-key

# WebSocket URL for the live inspection feed
VITE_WS_URL=ws://localhost:8000/ws/live
```

> **Critical:** `VITE_API_KEY` must be the **same string** as `API_KEY` in the root `.env`. If they differ, every API call will return `401 Unauthorized`.

---

## Step 4 — Python Virtual Environment

```powershell
# From project root d:\Capstone Project code\Capstone-Project
python -m venv .venv

# Activate the venv
.\.venv\Scripts\Activate.ps1
# You should now see (.venv) prefix in your prompt
```

If PowerShell blocks script execution:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

---

## Step 5 — Install Python Dependencies

```powershell
# With venv activated:
pip install --upgrade pip

# Core runtime dependencies (install these for running the app)
pip install motor pymongo pydantic-settings fastapi uvicorn sqlalchemy aiosqlite httpx python-multipart

# Then install the full requirements for completeness:
pip install -r requirements.txt
```

> **Why two steps?** `requirements.txt` includes heavy ML packages (PyTorch, ONNX, OpenCV). If you only want the API + dashboard working without inference, the first `pip install` line is sufficient.

### Missing packages — add these (not yet in requirements.txt):

```powershell
pip install motor pymongo
```

`motor` (async MongoDB driver) and `pymongo` (sync MongoDB driver, needed internally by motor) are required for the new Products/Runs endpoints but are **not yet listed in `requirements.txt`**. This is a known gap — they need to be added.

After confirming they work, add to `requirements.txt`:
```
motor==3.5.1
pymongo==4.8.0
```

---

## Step 6 — Start the Backend API

```powershell
# From project root, with .venv activated
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

Expected healthy startup output:
```
INFO:     Started server process [XXXX]
INFO:     Waiting for application startup.
INFO:     Motor client initialised — db: visionfood
INFO:     MongoDB indexes created.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

If you see instead:
```
RuntimeError: Motor database is not initialised.
```
→ MongoDB is not running. Start the MongoDB service (Step 2).

**Verify the API is healthy:**
```powershell
# Open in browser or run in PowerShell:
Invoke-RestMethod -Uri "http://localhost:8000/health" -Headers @{"X-API-Key"="dev-insecure-key"}
```

**Interactive API docs (Swagger UI):**
```
http://localhost:8000/docs
```

---

## Step 7 — (Optional) Install and Start Redis

Redis is needed for the **Live tab** WebSocket feed. Without it, the Live tab will show offline, but all other tabs work.

**Option A — Windows (via Memurai, a Redis-compatible Windows build):**
```
Download: https://www.memurai.com/get-memurai
```

**Option B — Docker:**
```powershell
docker run -d --name visionfood-redis -p 6379:6379 redis:7-alpine
```

**Verify:**
```powershell
# With Docker running:
docker exec visionfood-redis redis-cli ping
# Should respond: PONG
```

---

## Step 8 — Install Frontend Dependencies and Start Dashboard

```powershell
cd dashboard
npm install
npm run dev
```

Expected output:
```
VITE v5.x.x  ready in 300ms

➜  Local:   http://localhost:3000/
➜  Network: use --host to expose
```

Open `http://localhost:3000` in your browser.

> **Make sure the backend (Step 6) is running before opening the dashboard.** Vite proxies all `/api` requests to `localhost:8000`. If the backend is not running, all API calls will fail with `Failed to fetch`.

---

## Step 9 — Verify Everything Is Connected

### 9.1 Login Screen
The dashboard loads a login page. Use any username/password (auth is currently header-based, not validated on the frontend beyond a non-empty check). After "logging in", you land on the main dashboard.

### 9.2 Settings Tab (Product Registration)
1. Click the **Settings** tab (visible to `supervisor` and `admin` roles).
2. If the backend is healthy, the **SKU Profile** dropdown in the `ProductRegistration` form should populate with the YAML files from `configs/sku_profiles/` (you should see `bottle_250ml`, `can_330ml`, `pouch_100g`, `lays_50g`, etc.).
3. Fill in a test product and click **Register Product** → should return a green success toast.

### 9.3 Live Tab (Run Setup)
1. Click the **Live** tab.
2. The `RunSetup` banner at the top should show **"No active production run"** and a **Start Run** button.
3. Click **Start Run** → the SKU dropdown should populate with your registered products from Step 9.2.

### 9.4 API Smoke Test (via Swagger or PowerShell)
```powershell
# List all products
Invoke-RestMethod `
  -Uri "http://localhost:8000/api/v1/products" `
  -Headers @{"X-API-Key"="dev-insecure-key"}

# List active run
Invoke-RestMethod `
  -Uri "http://localhost:8000/api/v1/runs/active" `
  -Headers @{"X-API-Key"="dev-insecure-key"}
```

---

## Step 10 — Database Reset (if needed)

### Reset SQLite (inspection logs):
```powershell
# From project root:
Remove-Item visionfood_dev.db -ErrorAction SilentlyContinue
# The file is auto-recreated on next API startup
```

### Reset MongoDB (products + runs):
Open MongoDB Compass → connect to `mongodb://localhost:27017` → select database `visionfood` → drop the `products` and `production_runs` collections.

Or via PowerShell:
```powershell
# Requires mongosh installed
mongosh visionfood --eval "db.products.drop(); db.production_runs.drop()"
```

---

## Known Issues & Workarounds

| Issue | Root Cause | Workaround |
|---|---|---|
| `motor` not found on import | Not in `requirements.txt` yet | `pip install motor pymongo` manually |
| `mongomock-motor` not found (tests only) | Dev dependency not listed | `pip install mongomock-motor` |
| Products API returns 500 on startup if MongoDB is down | Motor init fails silently | Start MongoDB before starting the backend |
| Live tab shows "Offline" | Redis not running | Install Redis (Step 7) or ignore for now |
| Duplicate type declarations in `types.ts` | File has `BoundingBox`, `Detection`, etc. defined twice | TypeScript ignores duplicates for identical types — no runtime error, but needs cleanup |
| `core/schemas.py` missing V2 enums (`ProductSubType`, etc.) | Collaborator A has not added them yet | API uses internal string-literal validation sets — fully functional, just not canonical enums |
| `pytest-benchmark` not in requirements-dev.txt | Missing dependency for benchmarks | `pip install pytest-benchmark` |

---

## Summary Checklist

```
[ ] MongoDB installed and running on port 27017
[ ] Copy .env.example → .env (project root) and fill in MONGO_URL
[ ] Copy dashboard/.env.example → dashboard/.env
[ ] Python 3.11 venv created and activated
[ ] pip install motor pymongo (missing from requirements.txt)
[ ] pip install -r requirements.txt
[ ] uvicorn api.main:app --reload (backend on port 8000)
[ ] cd dashboard && npm install && npm run dev (frontend on port 3000)
[ ] Open http://localhost:3000 and verify SKU profile dropdown loads in Settings
[ ] (Optional) Install Redis for live WebSocket feed
```
