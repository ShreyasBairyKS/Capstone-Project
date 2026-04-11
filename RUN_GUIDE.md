# VisionFood QAI — Run Guide

---

## Prerequisites

| Tool | Version | Install |
|------|---------|---------|
| Python | 3.11.x | [python.org](https://www.python.org/downloads/) or Miniconda |
| Node.js | 20.x LTS | [nodejs.org](https://nodejs.org/) |
| Git | Any recent | `git --version` |
| Redis | 7.x | Docker (recommended) or native install |
| CUDA + cuDNN | 12.1+ / 8.x | If using GPU inference (optional) |
| Docker Desktop | Latest | [docker.com](https://www.docker.com/products/docker-desktop/) — for full stack |

GPU is optional. The full pipeline runs on CPU with ONNX Runtime (slower but functional for development and demo).

---

## Option A — Quick Start (Docker Compose)

The fastest way to get the full stack running.

```bash
# 1. Clone repository
git clone https://github.com/<your-org>/visionfood-qai.git
cd visionfood-qai

# 2. Copy environment config
cp .env.example .env
# Edit .env — minimum required: API_KEY, MODEL_PATHS (can be defaults)

# 3. Place trained models in models/ directory
#    models/yolov11n_best.onnx
#    models/efficientvit_m5_best.onnx

# 4. Build and run all services
docker compose up --build

# Services started:
#   API backend:   http://localhost:8000
#   Dashboard:     http://localhost:3000
#   API docs:      http://localhost:8000/docs
#   Redis:         localhost:6379
#   MLflow UI:     http://localhost:5000
```

To stop: `docker compose down`  
To rebuild after code change: `docker compose up --build --force-recreate`

---

## Option B — Manual Setup (Development)

### Step 1 — Python Environment

```bash
# Create and activate conda environment
conda create -n visionfood python=3.11
conda activate visionfood

# Install ONNX Runtime (choose one):
pip install onnxruntime          # CPU only
pip install onnxruntime-gpu      # CUDA GPU (requires CUDA 12.1)

# Install all dependencies
pip install -r requirements.txt

# Verify core imports
python -c "import torch, ultralytics, timm, fastapi, onnxruntime; print('All OK')"
```

### Step 2 — Environment Configuration

```bash
cp .env.example .env
```

Edit `.env`:
```env
# Required
API_KEY=your_secret_key_here
DATABASE_URL=sqlite:///./visionfood_dev.db
REDIS_URL=redis://localhost:6379

# Model paths (relative to project root)
YOLOV11_ONNX_PATH=models/yolov11n_best.onnx
EFFICIENTVIT_ONNX_PATH=models/efficientvit_m5_best.onnx

# Camera
CAMERA_INDEX=0           # 0 = default webcam, 1 = second camera
CAMERA_MODE=software     # "software" (spacebar trigger) or "hardware" (GPIO)

# Inference thresholds
YOLOV11_CONF_THRESHOLD=0.40
AUTO_PASS_THRESHOLD=0.85
ESCALATE_THRESHOLD=0.60
HUMAN_REVIEW_THRESHOLD=0.45

# Optional performance/logging
LOG_LEVEL=INFO
LOG_FORMAT=json
REMEDY_ENABLED=true
```

### Step 3 — Database Setup

```bash
# Run Alembic migrations (creates tables in SQLite)
alembic upgrade head

# (Optional) Seed with sample data for dashboard development
python scripts/seed_db.py --records 500
```

### Step 4 — Start Redis

```bash
# Using Docker (easiest):
docker run -d -p 6379:6379 --name visionfood-redis redis:7-alpine

# Or if Redis is installed natively:
redis-server
```

### Step 5 — Start Celery Worker (for PDF reports)

```bash
# In a separate terminal:
celery -A api.celery_app worker --loglevel=info -Q reports
```

### Step 6 — Start Backend API

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

Expected output:
```
INFO:     Started server process [12345]
INFO:     Waiting for application startup.
INFO:     Loading YOLOv11 ONNX model...
INFO:     Loading EfficientViT ONNX model...
INFO:     Models loaded. Warmup complete.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

Visit `http://localhost:8000/health` — should return `{"status": "ok", "model_loaded": true, ...}`

### Step 7 — Start Dashboard

```bash
cd dashboard
npm install
npm run dev
# Dashboard opens at http://localhost:3000
# Proxies /api/* to http://localhost:8000 automatically (configured in vite.config.ts)
```

---

## Training the Models

### Train YOLOv11 Detector

```bash
# Activate environment
conda activate visionfood

# Place your annotated dataset at: data/annotated/
# Verify data.yaml exists at: data/annotated/data.yaml

# Training (takes 30–90 minutes on GPU, up to 8 hours on CPU)
python training/train_yolov11.py

# Or directly with ultralytics CLI:
yolo train \
  model=yolov11n.pt \
  data=data/annotated/data.yaml \
  epochs=100 \
  imgsz=640 \
  batch=16 \
  project=runs/train \
  name=yolov11n_visionfood_v1

# Best model saved to: runs/train/yolov11n_visionfood_v1/weights/best.pt
```

### Export YOLOv11 to ONNX

```bash
yolo export model=runs/train/yolov11n_visionfood_v1/weights/best.pt \
  format=onnx half=True imgsz=640

# Copy to models/
cp runs/train/yolov11n_visionfood_v1/weights/best.onnx models/yolov11n_best.onnx
```

### Train EfficientViT Classifier

```bash
# Prepare crop dataset first (extracts bbox crops from annotated images)
python scripts/prepare_crop_dataset.py \
  --annotations data/annotated/ \
  --output data/crops/ \
  --padding 0.10

# Train classifier
python training/train_efficientvit.py \
  --data data/crops/ \
  --epochs 50 \
  --batch 32 \
  --output models/efficientvit_m5_best.pt

# Export to ONNX
python export/export_onnx.py --model efficientvit
# Output: models/efficientvit_m5_best.onnx
```

### Evaluate Models

```bash
# Evaluate YOLOv11 on test set
python training/evaluate_yolov11.py \
  --model models/yolov11n_best.onnx \
  --data data/annotated/data.yaml \
  --split test \
  --output results/

# Benchmark inference latency
python scripts/benchmark_inference.py \
  --model models/yolov11n_best.onnx \
  --frames 200 \
  --device cpu

# Output: p50, p95, p99 latency + throughput (products/min)
```

---

## Running Inspection from Command Line

```bash
# Inspect a single image
python -m inference.run_single --image test_data/bottle_defect.jpg

# Live camera inspection loop (press Q to quit)
python -m inference.capture_loop --camera 0 --mode software

# Batch inspect a folder of images
python -m inference.run_batch --input test_data/ --output results/batch_results.json
```

---

## Running Tests

```bash
# All tests
pytest tests/ -v

# Unit tests only
pytest tests/unit/ -v

# Integration tests (requires running API + database)
pytest tests/integration/ -v

# With coverage report
pytest tests/ --cov=. --cov-report=html
open htmlcov/index.html   # View coverage report
```

---

## MLflow Experiment Tracking

```bash
# Start MLflow UI
mlflow ui --host 0.0.0.0 --port 5000

# View at: http://localhost:5000
# All training runs, metrics, and model artefacts visible here
```

---

## Common Issues

| Issue | Cause | Fix |
|-------|-------|-----|
| `onnxruntime` import error | Wrong package installed | `pip uninstall onnxruntime; pip install onnxruntime-gpu` |
| `Model not loaded` on `/health` | ONNX file path wrong | Check `YOLOV11_ONNX_PATH` in `.env` |
| Redis connection refused | Redis not running | `docker run -d -p 6379:6379 redis:7-alpine` |
| Camera index 0 not found | Wrong camera index | Try `CAMERA_INDEX=1` in `.env` |
| Alembic error on `upgrade head` | `DATABASE_URL` not set | Set `DATABASE_URL=sqlite:///./visionfood_dev.db` in `.env` |
| `CUDA out of memory` during training | Batch too large for VRAM | Reduce `batch=8` in training command |
| Dashboard blank page | API CORS issue | Set `CORS_ORIGINS=http://localhost:3000` in `.env` |
| PDF report stuck at `pending` | Celery worker not running | Start with `celery -A api.celery_app worker` |

---

## Directory Layout After Full Setup

```
visionfood-qai/
├── models/
│   ├── yolov11n_best.pt         ← PyTorch checkpoint
│   ├── yolov11n_best.onnx       ← ONNX (deployed)
│   ├── efficientvit_m5_best.pt
│   ├── efficientvit_m5_best.onnx
│   └── active_version.json      ← {"version": "yolov11n_v1.0.0", ...}
├── data/
│   ├── annotated/               ← Roboflow export (train/val/test + data.yaml)
│   └── crops/                   ← EfficientViT training crops
├── runs/
│   └── train/                   ← YOLOv11 training runs
├── results/                     ← Evaluation outputs
├── visionfood_dev.db            ← SQLite database (dev)
└── logs/
    └── audit.jsonl              ← Append-only audit log
```

---

## Phase 7–9 Production Features

### Prometheus Metrics

Once Phase 7 is implemented, metrics are available at:
```bash
curl http://localhost:8000/metrics
# Returns Prometheus text exposition format

# Example metrics:
# visionfood_http_requests_total{method="POST",path="/inspections",status="201"} 1247
# visionfood_inference_duration_seconds_sum{method="POST",path="/inspections"} 52.34
# visionfood_inspection_verdicts_total{verdict="PASS"} 1050
# visionfood_model_loaded{model="yolov11n"} 1
```

### Readiness Probe
```bash
curl http://localhost:8000/readiness
# Returns: model loaded, DB reachable, Redis reachable, system info
```

### Batch Inspection
```bash
# Inspect multiple images in one request
curl -X POST http://localhost:8000/inspections/batch \
  -H "X-API-Key: your_key" \
  -H "Content-Type: application/json" \
  -d '{"images": [{"image_b64": "...", "sku": "bottle_250ml"}, ...]}'
```

### Performance Benchmarking
```bash
# Benchmark with synthetic frames
python scripts/benchmark_inference.py --frames 200 --device cpu

# Benchmark with test images
python scripts/benchmark_inference.py --input test_data/ --device cuda

# Output example:
#   p50 latency:   38.2 ms
#   p95 latency:   52.1 ms
#   p99 latency:   67.4 ms
#   Throughput:    26.2 items/sec
#   Memory RSS:   312 MB
```

### TensorRT Export (Jetson Edge)
```bash
# Export YOLOv11 to TensorRT (requires TensorRT installed)
python export/export_tensorrt.py --model detector --precision int8

# Export EfficientViT to TensorRT
python export/export_tensorrt.py --model classifier --precision fp16

# Output: models/yolov11n_best.engine, models/efficientvit_m5_best.engine
```

### Grad-CAM++ Explainability
```bash
# Inspect with visual explanation
curl -X POST "http://localhost:8000/inspections?explain=true" \
  -H "X-API-Key: your_key" \
  -H "Content-Type: application/json" \
  -d '{"image_b64": "...", "sku": "default"}'

# Response includes "explanation.heatmap_b64" — base64 PNG heatmap overlay
```

### Production Docker Compose
```bash
# Production stack with PostgreSQL, Prometheus, Grafana
docker compose -f docker/docker-compose.prod.yml up --build

# Services:
#   API:          http://localhost:8000
#   Dashboard:    http://localhost:3000  (via NGINX + TLS)
#   PostgreSQL:   localhost:5432
#   Redis:        localhost:6379
#   MLflow:       http://localhost:5000
#   Prometheus:   http://localhost:9090
#   Grafana:      http://localhost:3001
```

### CI/CD
```bash
# CI runs automatically on push/PR to dev/main branches
# Manual deploy to production:
gh workflow run deploy.yml -f environment=production
```
