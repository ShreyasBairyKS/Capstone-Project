# VisionFood QAI
### Intelligent Quality Inspection System for Food & Beverage Manufacturing

> **Capstone Project — AI/Deep Learning**  
> An automated defect detection pipeline that inspects food and beverage packaging using CNNs and object detection, generates quality reports, and flags defects in real time through a software inspection interface.

---

## Problem Statement

Manual quality inspection in food manufacturing is slow, inconsistent, and expensive. This system replaces manual inspection with a deep learning pipeline that detects and classifies four defect categories — **improper filling, packaging damage, label misalignment, and surface contamination** — from product images captured during manufacturing.

---

## What Is Actually Built

| Component | Status | Technology |
|-----------|--------|------------|
| YOLOv11 defect detector | Built (code) — awaiting dataset | Ultralytics, ONNX |
| EfficientViT-M5 classifier | Built (code) — awaiting dataset | timm, ONNX |
| Uncertainty quantification | Built | MC Dropout |
| REMEDY severity engine | Built (software) | Python |
| FastAPI inspection backend | Built | FastAPI, SQLite |
| Real-time dashboard | Built | React 18, Vite, Tailwind, Recharts |
| Quality report generator | Built | ReportLab PDF |
| Camera capture loop | Built | OpenCV, webcam |
| ONNX export utility | Built | torch.onnx, Ultralytics export |
| Test suite (100 tests) | Passing | pytest, httpx, anyio |
| Prometheus metrics middleware | Built | Zero-dependency, text exposition |
| Batch inspection endpoint | Planned (Phase 7) | FastAPI, async |
| Rate limiter middleware | Planned (Phase 7) | Token-bucket, in-memory |
| Health & readiness probes | Planned (Phase 7) | Kubernetes-compatible |
| Grad-CAM++ explainability | Planned (Phase 8) | PyTorch hooks |
| Drift detection | Planned (Phase 8) | KL divergence |
| TensorRT export | Planned (Phase 8) | TensorRT, ARM64/x86 |
| CI/CD pipeline | Planned (Phase 9) | GitHub Actions |
| Production Docker stack | Planned (Phase 9) | Docker Compose, NGINX TLS |

> **Extended Architecture** (designed, not physically built): multi-sensor fusion (FLIR, NIR, DVS), Jetson TensorRT deployment, HAFFN fog fusion, CDAG-Net causal AI, Mamba-QC forecasting, federated learning, and hardware REMEDY stations. Full design is documented in `VisionFood_QAI_System_Architecture.md`.

---

## Defect Classes

| Class | Description |
|-------|-------------|
| `improper_filling` | Underfill, overfill, visible air gaps |
| `packaging_damage` | Dents, cracks, tears, seal failures |
| `label_misalignment` | Skew, wrinkle, missing label, offset |
| `surface_contamination` | Stains, mould spots, foreign particles |

---

## Project Phases (18-Week Plan)

| Phase | Title | Weeks | Status | Deliverable |
|-------|-------|-------|--------|-------------|
| **0** | Environment & Dataset | 1–2 | ✅ Complete | Annotated dataset, project scaffold |
| **1** | YOLOv11 Training | 3–4 | ✅ Complete | Trained detector, mAP ≥ 0.80 |
| **2** | Classifier + Pipeline | 5 | ✅ Complete | End-to-end inference pipeline, ONNX export |
| **3** | REMEDY Engine | 6 | ✅ Complete | Severity scorer, triage router |
| **4** | Backend & Database | 7–8 | ✅ Complete | FastAPI server, inspection records, reporting endpoints |
| **5** | Dashboard & Reports | 9–10 | ✅ Complete | React dashboard, PDF quality reports |
| **6** | Integration & Demo | 11–12 | ✅ Complete | Full system demo, final documentation |
| **7** | Production Hardening | 13–14 | 🔄 In Progress | Metrics, batch API, rate limiter, health probes |
| **8** | Explainability & Edge | 15–16 | 🔲 Planned | Grad-CAM++ XAI, drift detection, TensorRT export |
| **9** | CI/CD & Final Polish | 17–18 | 🔲 Planned | GitHub Actions, production Docker, NGINX TLS, CHANGELOG |

---

## Repository Structure

```
visionfood-qai/
│
├── core/                    # Shared config, schemas, logging
├── inference/               # YOLOv11 detector, EfficientViT classifier, UQ
│   └── explainability/      # Grad-CAM++ heatmap generation (Phase 8)
├── remedy/                  # Severity scorer, triage router
├── api/                     # FastAPI backend + WebSocket
│   └── middleware/          # Auth, audit logger, metrics, rate limiter
├── dashboard/               # React frontend
├── database/                # SQLAlchemy models, migrations
├── training/                # Model training scripts
├── export/                  # ONNX / TensorRT export scripts
├── tests/                   # Unit, integration, e2e tests
├── reports/                 # PDF report templates
├── data/                    # Dataset (gitignored)
│   ├── raw/
│   ├── annotated/
│   └── splits/
├── models/                  # Trained model artefacts (gitignored)
├── configs/                 # YAML configs per SKU
├── docker/                  # Dockerfiles, docker-compose, docker-compose.prod
│   └── Dockerfile.edge      # ARM64/Jetson edge deployment (Phase 9)
├── scripts/                 # Setup, benchmark, calibration scripts
└── .github/workflows/       # CI/CD pipelines (Phase 9)
```

---

## Quick Start

```bash
# 1. Clone
git clone https://github.com/<your-org>/visionfood-qai.git
cd visionfood-qai

# 2. Create Python environment
conda create -n visionfood python=3.11
conda activate visionfood
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# Edit .env — set DATABASE_URL, model paths, camera index

# 4. Run database migrations
alembic upgrade head

# 5. Start backend
uvicorn api.main:app --reload --port 8000

# 6. Start dashboard
cd dashboard && npm install && npm start
```

> Full setup instructions: see [RUN_GUIDE.md](RUN_GUIDE.md)

---

## Technology Stack

| Layer | Technology |
|-------|------------|
| Object Detection | YOLOv11n (Ultralytics) |
| Classification | EfficientViT-M5 (timm) |
| Uncertainty | MC Dropout (PyTorch) |
| Edge Runtime | ONNX Runtime 1.18, TensorRT (Phase 8) |
| Explainability | Grad-CAM++ (Phase 8) |
| Backend | FastAPI 0.111, SQLite/PostgreSQL |
| Observability | Prometheus metrics, structured logging |
| Real-time | Redis Streams, WebSocket |
| Frontend | React, Recharts, WebSocket |
| Report Generation | ReportLab |
| Experiment Tracking | MLflow, Weights & Biases |
| Data Annotation | Roboflow |
| Containerisation | Docker, Docker Compose, NGINX |
| CI/CD | GitHub Actions (Phase 9) |
| Edge Deployment | ARM64 Docker, Jetson Orin (Phase 9) |

---

## Key Documents

| Document | Purpose |
|----------|---------|
| [ARCHITECTURE.md](ARCHITECTURE.md) | Capstone system architecture |
| [IMPLEMENTATION.md](IMPLEMENTATION.md) | Phase-by-phase implementation plan |
| [DATA_MODEL.md](DATA_MODEL.md) | Database schema and Pydantic models |
| [API_REFERENCE.md](API_REFERENCE.md) | All FastAPI endpoints |
| [ML_GRAPH_BACKEND_EXPLANATIONS.md](ML_GRAPH_BACKEND_EXPLANATIONS.md) | Deep learning pipeline walkthrough |
| [RUN_GUIDE.md](RUN_GUIDE.md) | Setup and run instructions |
| [TEAM_EXECUTION_GUIDE.md](TEAM_EXECUTION_GUIDE.md) | Team workflow and Git process |
| [ROLE_FILE_BREAKDOWN.md](ROLE_FILE_BREAKDOWN.md) | Role → file ownership map |
| [ROLE_HANDOFF_CHECKLIST.md](ROLE_HANDOFF_CHECKLIST.md) | Phase handoff checklists |
| [PROMPT_PLAYBOOK.md](PROMPT_PLAYBOOK.md) | AI-assisted coding prompts |

---

## Performance Targets

| Metric | Target |
|--------|--------|
| mAP@50 (YOLOv11) | ≥ 0.80 |
| Top-1 Accuracy (EfficientViT) | ≥ 95% |
| False Negative Rate | < 3% |
| Inference Latency (ONNX, laptop CPU) | < 80ms |
| Inference Latency (ONNX, CUDA GPU) | < 15ms |
| Inference Latency (TensorRT, Jetson) | < 25ms |
| Dashboard Refresh Rate | ≥ 10 fps (via WebSocket) |
| API p99 Latency | < 200ms |
| Batch Throughput | ≥ 10 images/sec |

---

*VisionFood QAI — Capstone Project, March 2026*
