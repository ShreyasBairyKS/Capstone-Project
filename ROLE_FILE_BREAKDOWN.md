# VisionFood QAI — Role & File Breakdown

---

## Role Definitions

| Role | Short Code | Primary Expertise |
|------|-----------|------------------|
| ML Engineer | `ML` | Deep learning, model training, ONNX optimisation |
| Backend Engineer | `BE` | FastAPI, databases, async systems, Redis |
| Frontend Engineer | `FE` | React, WebSocket, data visualisation |
| DevOps / All | `ALL` | Docker, CI/CD, environment, integration |

---

## File Ownership Map

> **Owner** = primary responsible person for implementation and code review.  
> **Reviewer** = must review any PR touching this file.

### `core/`

| File | Owner | Reviewer | Notes |
|------|-------|----------|-------|
| `core/config.py` | ALL | ALL | Shared — any change must be reviewed by all |
| `core/schemas.py` | BE | ML | Pydantic schemas used by both API and inference |
| `core/logging.py` | BE | — | Structured JSON logging setup |
| `core/messaging.py` | BE | — | Redis pub/sub wrappers |

---

### `inference/`

| File | Owner | Reviewer | Notes |
|------|-------|----------|-------|
| `inference/models/yolov11_detector.py` | ML | BE | ONNX inference wrapper for YOLOv11 |
| `inference/models/efficientvit_classifier.py` | ML | BE | ONNX classifier wrapper |
| `inference/models/uq_inspector.py` | ML | BE | MC Dropout + Deep Ensemble UQ |
| `inference/models/model_registry.py` | BE | ML | Active model pointer management |
| `inference/preprocessor.py` | ML | — | Letterbox resize, normalise, NCHW |
| `inference/postprocessor.py` | ML | — | NMS, denormalise, area fraction |
| `inference/pipeline.py` | ML | BE | Core orchestration — verdict logic |
| `inference/camera.py` | ML | — | OpenCV VideoCapture wrapper |
| `inference/capture_loop.py` | ML | BE | Async production capture loop |

---

### `remedy/`

| File | Owner | Reviewer | Notes |
|------|-------|----------|-------|
| `remedy/severity_scorer.py` | BE | ML | Severity formula — weights tuned by ML |
| `remedy/triage_router.py` | BE | ML | Action routing per (class, grade) |
| `remedy/sku_profile_manager.py` | BE | — | YAML profile loader |
| `remedy/reinspection_loop.py` | BE | ML | Re-inspection after remediation |
| `remedy/station_controller.py` | BE | — | PLC/Arduino bridge (stub for capstone) |

---

### `training/`

| File | Owner | Reviewer | Notes |
|------|-------|----------|-------|
| `training/train_yolov11.py` | ML | — | YOLOv11 fine-tuning script |
| `training/train_efficientvit.py` | ML | — | EfficientViT-M5 classifier training |
| `training/evaluate_yolov11.py` | ML | — | mAP, PR curve, confusion matrix |
| `training/active_learning.py` | ML | — | BADGE2 sample selection |
| `training/synthetic_augment.py` | ML | — | StyleGAN3 / albumentations augmentation |

---

### `export/`

| File | Owner | Reviewer | Notes |
|------|-------|----------|-------|
| `export/export_onnx.py` | ML | — | PyTorch → ONNX FP16 |
| `export/export_tensorrt.py` | ML | — | ONNX → TensorRT INT8 (Jetson only) |
| `export/export_openvino.py` | ML | — | ONNX → OpenVINO IR (Intel NUC) |
| `export/quantisation_aware_training.py` | ML | — | QAT fallback if INT8 accuracy drop |

---

### `api/`

| File | Owner | Reviewer | Notes |
|------|-------|----------|-------|
| `api/main.py` | BE | ALL | App factory + lifespan — critical |
| `api/dependencies.py` | BE | ALL | DB session + model injection |
| `api/routers/inspection.py` | BE | ML | POST /inspect — runs pipeline |
| `api/routers/analytics.py` | BE | FE | Analytics endpoints — schema agreed with FE |
| `api/routers/reports.py` | BE | FE | Report generation trigger |
| `api/routers/models.py` | BE | ML | Model version management |
| `api/routers/websocket.py` | BE | FE | WS /ws/live — coordinates with FE |
| `api/middleware/auth.py` | BE | — | API key authentication |
| `api/middleware/audit_logger.py` | BE | — | Append-only request log |

---

### `database/`

| File | Owner | Reviewer | Notes |
|------|-------|----------|-------|
| `database/models.py` | BE | ML | SQLAlchemy ORM — schema is critical |
| `database/repositories/inspection_repo.py` | BE | — | Data access layer |
| `database/repositories/analytics_repo.py` | BE | — | Aggregation queries |
| `database/migrations/` | BE | — | Alembic migrations — never edit manually |

---

### `reports/`

| File | Owner | Reviewer | Notes |
|------|-------|----------|-------|
| `reports/generator.py` | BE | FE | ReportLab PDF builder |
| `reports/templates/` | BE+FE | — | Report layout assets |

---

### `dashboard/src/`

| File / Directory | Owner | Reviewer | Notes |
|-----------------|-------|----------|-------|
| `pages/LiveFeed.jsx` | FE | BE | WebSocket consumer, most complex page |
| `pages/Analytics.jsx` | FE | BE | Charts consume analytics API |
| `pages/History.jsx` | FE | BE | Paginated inspection table |
| `pages/Reports.jsx` | FE | BE | PDF download via reports API |
| `pages/Models.jsx` | FE | BE | Model version management |
| `components/BoundingBoxCanvas.jsx` | FE | ML | Draws YOLO bbox accurately |
| `components/DefectRateChart.jsx` | FE | — | Recharts LineChart |
| `components/ParetoChart.jsx` | FE | BE | Bar + line overlay |
| `components/SeverityBadge.jsx` | FE | BE | S1/S2/S3/S4 colour badges |
| `components/AlertPanel.jsx` | FE | BE | High-severity cluster alert |
| `hooks/useWebSocket.js` | FE | BE | WS management + reconnect |
| `hooks/useInspections.js` | FE | — | React Query API hooks |
| `store/inspectionStore.js` | FE | — | Zustand live state slice |

---

### `configs/`

| File | Owner | Reviewer | Notes |
|------|-------|----------|-------|
| `configs/edge_config.yaml` | BE | ML | Edge tier runtime config |
| `configs/sku_profiles/bottle_250ml.yaml` | BE | ML | Severity weights for bottles |
| `configs/sku_profiles/pouch_100g.yaml` | BE | ML | Severity weights for pouches |
| `configs/sku_profiles/can_330ml.yaml` | BE | ML | Severity weights for cans |

---

### `tests/`

| Directory | Owner | Notes |
|-----------|-------|-------|
| `tests/unit/test_config.py` | ALL | CI must always pass |
| `tests/unit/test_yolov11_detector.py` | ML | Phase 1 exit criterion |
| `tests/unit/test_severity_scorer.py` | BE | Phase 3 exit criterion |
| `tests/unit/test_triage_router.py` | BE | Phase 3 exit criterion |
| `tests/integration/test_pipeline_e2e.py` | ML | Phase 2 exit criterion |
| `tests/integration/test_api.py` | BE | Phase 4 exit criterion |
| `tests/e2e/test_full_pipeline.py` | ALL | Phase 6 exit criterion |

---

### `docker/` and CI

| File | Owner | Reviewer |
|------|-------|----------|
| `docker/Dockerfile.api` | BE | ALL |
| `docker/Dockerfile.dashboard` | FE | ALL |
| `docker/docker-compose.yml` | BE | ALL |
| `.github/workflows/ci.yml` | ALL | ALL |

---

## Review Rules Summary

- Any change to `core/config.py` or `core/schemas.py` requires **all team members** to review
- Any change to `inference/pipeline.py` verdict logic requires **ML + BE** review
- Any change to `database/models.py` must be accompanied by a new Alembic migration
- Frontend changes to API-consuming components require **FE + BE** review (schema contract)
- Test files are owned by the same role as the module they test, but **all team members** run them in Phase 6
