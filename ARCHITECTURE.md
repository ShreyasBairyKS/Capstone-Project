# VisionFood QAI — System Architecture

---

## Capstone Build vs Full Design

This document describes **two levels** of architecture:
- **[BUILD]** — Components that are physically implemented in this capstone.
- **[DESIGN]** — Components documented in the extended system architecture (`VisionFood_QAI_System_Architecture.md`) but not built as part of the capstone submission.

All [DESIGN] components are fully specified and the codebase is structured to accommodate them — they are genuine extensions, not padding.

---

## High-Level Architecture (Capstone Build)

```
┌─────────────────────────────────────────────────────────────────┐
│                     TIER 1 — CAPTURE [BUILD]                    │
│                                                                 │
│   USB Webcam / Camera  ─→  OpenCV Capture Loop                  │
│   (640×640 RGB frames, hardware or software trigger)            │
└────────────────────────────────┬────────────────────────────────┘
                                 │  numpy frame (640×640×3)
┌────────────────────────────────▼────────────────────────────────┐
│               TIER 2 — INFERENCE ENGINE [BUILD]                 │
│                                                                 │
│  ┌─────────────────┐    ┌──────────────────┐   ┌─────────────┐ │
│  │  YOLOv11n       │ →  │ EfficientViT-M5  │ → │ MC Dropout  │ │
│  │  ONNX Detector  │    │ ONNX Classifier  │   │ UQ Module   │ │
│  │  (640×640 in)   │    │  (crop in)       │   │ (20 passes) │ │
│  └─────────────────┘    └──────────────────┘   └─────────────┘ │
│                                                                 │
│        PASS / FAIL / ESCALATE / REVIEW verdict                  │
│                             │                                   │
│              ┌──────────────▼──────────────┐                    │
│              │   REMEDY Severity Engine    │                    │
│              │   (software-only)           │                    │
│              │   S1/S2 → remediation log   │                    │
│              │   S3/S4 → reject            │                    │
│              └─────────────────────────────┘                    │
└────────────────────────────────┬────────────────────────────────┘
                                 │  InspectionResult (JSON)
┌────────────────────────────────▼────────────────────────────────┐
│              TIER 3 — BACKEND & DATABASE [BUILD]                │
│                                                                 │
│   FastAPI ─→ SQLite/PostgreSQL   Redis Streams (live push)      │
│             (inspection records)  (WebSocket fan-out)           │
│                                                                 │
│   Endpoints: /inspect  /analytics  /reports  /models           │
└────────────────────────────────┬────────────────────────────────┘
                                 │  REST + WebSocket
┌────────────────────────────────▼────────────────────────────────┐
│               TIER 4 — DASHBOARD & REPORTS [BUILD]              │
│                                                                 │
│   React SPA  →  Live feed with bounding box overlay             │
│                 Defect analytics (Recharts)                     │
│                 Inspection history table                        │
│                 PDF quality report download                     │
└─────────────────────────────────────────────────────────────────┘
```

---

## Extended Architecture (Design Only)

The following tiers are fully specified in the extended system architecture document and are structured as genuine engineering extensions:

| Tier | Component | Reason Not Built |
|------|-----------|-----------------|
| Tier 1 extended | FLIR Lepton 3.5, Hamamatsu NIR, RealSense D435i, DAVIS346 DVS | Hardware cost (₹15–20L), not required for core ML objective |
| Tier 2 extended | TensorRT INT8 on Jetson Orin NX, SNN event processor | Jetson hardware not available |
| Tier 3 fog | HAFFN multi-modal fusion, RT-DETRv2, SAM 2, Depth Anything V2 | Requires multi-sensor data from Tier 1 extended |
| Tier 4 cloud | DeFCNet severity regression, MAE+PatchCore anomaly detection | Training infra; add-on when base pipeline is stable |
| Tier 5 intelligence | CDAG-Net causal AI, Mamba-QC forecasting, SPC engine | Requires production historian data |
| Tier 7 | Federated learning (FedProx + EWC + DP-SGD) | Requires multi-site deployment |
| REMEDY hardware | Station A (relabelling), B (refill), C (repackaging) | Physical hardware not in scope |

---

## Component Architecture Detail

### Inference Engine (Tier 2)

```
Camera Frame (BGR, 640×640)
        │
        ▼
   Preprocessor
   ├── BGR → RGB conversion
   ├── Letterbox resize to 640×640
   ├── Normalise [0, 255] → [0.0, 1.0]
   └── HWC → NCHW, add batch dim
        │
        ▼
   YOLOv11n (ONNX Runtime)
   ├── Output: (1, 8400, 6) — 8400 anchor proposals
   │   Each: [x, y, w, h, confidence, class_id]
   ├── Confidence threshold filter: conf ≥ 0.40
   └── NMS IoU threshold: 0.45
        │
        ▼ List[Detection(class, conf, bbox, area_fraction)]
        │
   ┌────┴───────────────────────────────────┐
   │ No detections?                         │
   │   └── Verdict = PASS, skip step below  │
   └────┬───────────────────────────────────┘
        │ Top detection bounding box
        ▼
   Crop + 10% padding
        │
        ▼
   EfficientViT-M5 (ONNX Runtime)
   ├── 4-class softmax output
   └── Focal loss trained, AdamW
        │
        ▼
   MCDropoutUQ (20 forward passes, dropout active)
   ├── mean_confidence  (μ)
   ├── std_confidence   (σ)
   └── confidence_interval [μ-2σ, μ+2σ]
        │
        ▼
   Verdict Logic
   ├── conf ≥ 0.85 AND σ < 0.12  →  PASS or FAIL (final)
   ├── 0.60 ≤ conf < 0.85         →  FAIL + escalation flag
   ├── 0.45 ≤ conf < 0.60         →  ESCALATE (human review)
   └── conf < 0.45                →  REVIEW (manual only)
        │
        ▼
   REMEDY Severity Scorer (if FAIL)
   ├── S1/S2 → log remediation_action, mark as "remediated pass"
   └── S3/S4 → hard reject
```

### REMEDY Engine (Software)

```
SeverityScorer.score(detection, attempt_count)
│
├── area_score   = min(1.0, detection.area_fraction × 20)
├── conf_score   = 1.0 - detection.confidence
├── class_score  = CLASS_SEVERITY_WEIGHTS[class_name]
│     contamination=0.90, damage=0.60, fill=0.45, label=0.30
└── attempt_score = min(1.0, attempt_count × 0.30)
│
raw_score = 0.35×area + 0.15×conf + 0.40×class + 0.10×attempt
│
├── raw_score < 0.30  →  S1 (Minor, remediable)
├── 0.30–0.55         →  S2 (Moderate, remediable)
├── 0.55–0.75         →  S3 (Serious, reject)
└── ≥ 0.75            →  S4 (Critical, hard reject)

TriageRouter maps (class, grade) → action:
  label_misalignment + S1/S2  → RELABEL  (logged)
  improper_filling   + S1/S2  → REFILL   (logged)
  packaging_damage   + S1/S2  → REPACK   (logged)
  surface_contamination + S1  → CLEAN    (logged)
  anything S3/S4              → REJECT
```

### Backend Architecture

```
api/
├── main.py                   ← FastAPI app, CORS, middleware, lifespan
├── routers/
│   ├── inspection.py         ← POST /inspect, GET /inspections/{id}
│   ├── analytics.py          ← GET /analytics/summary, /defect-rate
│   ├── reports.py            ← POST /reports/generate, GET /reports/{id}
│   ├── models.py             ← GET /models, PUT /models/{id}/activate
│   └── websocket.py          ← WS /ws/live
├── middleware/
│   ├── auth.py               ← API key authentication
│   └── audit_logger.py       ← Append-only request logging
└── dependencies.py           ← DB session, model loader injection

Database: SQLite (development) / PostgreSQL (production)
Cache/Streaming: Redis Streams (key: inspections:live)
Task queue: Celery (PDF report generation, async)
```

### Dashboard Architecture

```
dashboard/src/
├── pages/
│   ├── LiveFeed.jsx         ← WebSocket consumer, bounding box canvas overlay
│   ├── Analytics.jsx        ← Recharts: defect rate, Pareto, trend
│   ├── History.jsx          ← Paginated inspection table with filters
│   ├── Reports.jsx          ← Report generator + download
│   └── Models.jsx           ← Model version management
├── components/
│   ├── BoundingBoxCanvas.jsx ← Draws YOLO bbox + label + confidence on canvas
│   ├── DefectRateChart.jsx
│   ├── ParetoChart.jsx
│   ├── SeverityBadge.jsx
│   └── AlertPanel.jsx
├── hooks/
│   ├── useWebSocket.js      ← WS connection management + reconnect
│   └── useInspections.js    ← React Query API hooks
└── store/
    └── inspectionStore.js   ← Zustand slice for live state
```

---

## Data Flow — Single Product Inspection

```
1. Camera trigger (GPIO or software)
2. Frame captured → inference/pipeline.py::EdgeInferencePipeline.inspect()
3. YOLOv11 detects defects (1.5–80ms depending on hw)
4. If detections exist → EfficientViT classifies top detection
5. MC Dropout UQ computes confidence interval
6. Verdict assigned based on confidence thresholds
7. If FAIL → SeverityScorer grades severity → TriageRouter assigns action
8. InspectionResult written to SQLite via SQLAlchemy
9. Result published to Redis Stream "inspections:live"
10. WebSocket server pushes to all connected dashboard clients
11. Dashboard renders bounding box + verdict + badge in <100ms
12. Periodic analytics re-computed per batch/shift
```

---

## Technology Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Detection model | YOLOv11n | Fastest YOLO generation, native ONNX export, excellent on small datasets |
| Classifier | EfficientViT-M5 | Linear attention — faster than ViT while retaining accuracy |
| Inference runtime | ONNX Runtime | Hardware-agnostic; runs on CPU+GPU+Hailo without code change |
| UQ method | MC Dropout | No additional models needed; 20-pass variance sufficient for 4-class problem |
| Backend | FastAPI | Async-native, automatic OpenAPI docs, Pydantic integration |
| Database | SQLite → PostgreSQL | SQLite zero-config for dev; swap via DATABASE_URL env variable |
| Real-time | Redis Streams + WS | Decoupled producer/consumer; scales to multiple dashboard clients |
| Report generation | ReportLab | Pure Python, no external service dependency |
| Annotation | Roboflow | Version control + augmentation + YOLO format export in one tool |

---

## Production Hardening Architecture (Phase 7–9)

The following components extend the capstone build into a production-grade, edge-deployable system:

### Observability Stack

```
┌──────────────────────────────────────────────────────────┐
│                 OBSERVABILITY [Phase 7]                   │
│                                                          │
│  api/middleware/metrics.py                                │
│  ├── PrometheusMiddleware (HTTP request tracking)        │
│  ├── visionfood_http_requests_total (counter)            │
│  ├── visionfood_http_request_duration_seconds (histogram)│
│  ├── visionfood_inspection_verdicts_total (counter)      │
│  ├── visionfood_inference_duration_seconds (histogram)   │
│  ├── visionfood_model_loaded (gauge)                     │
│  └── visionfood_active_ws_connections (gauge)            │
│                                                          │
│  GET /metrics → Prometheus text exposition format        │
│  GET /health  → Liveness probe (uptime, memory)          │
│  GET /readiness → Readiness probe (model, DB, Redis)     │
│                                                          │
│  api/middleware/rate_limiter.py                           │
│  └── Sliding window per API key (100 RPM default)        │
│                                                          │
│  api/middleware/audit_logger.py [existing]                │
│  └── JSONL append-only audit trail                       │
└──────────────────────────────────────────────────────────┘
```

### Explainability & Drift Detection

```
┌──────────────────────────────────────────────────────────┐
│            EXPLAINABILITY & MONITORING [Phase 8]          │
│                                                          │
│  inference/explainability/gradcam.py                     │
│  ├── Grad-CAM++ heatmap for EfficientViT classifier     │
│  ├── Hooks into last convolutional layer                 │
│  └── Returns blended overlay (image + heatmap)           │
│                                                          │
│  inference/drift_detector.py                             │
│  ├── KL divergence between baseline & current distrib.   │
│  ├── Sliding window (last 500 inspections)               │
│  └── Alert when divergence > threshold (0.1)             │
│                                                          │
│  GET /analytics/drift → drift metrics + alert status     │
│  POST /inspections?explain=true → includes heatmap       │
└──────────────────────────────────────────────────────────┘
```

### Edge Deployment Architecture

```
┌──────────────────────────────────────────────────────────┐
│              EDGE DEPLOYMENT [Phase 8-9]                  │
│                                                          │
│  export/export_tensorrt.py                               │
│  ├── ONNX → TensorRT INT8/FP16 engine builder            │
│  ├── Calibration dataset from validation images           │
│  └── Target: Jetson Orin NX (ARM64)                      │
│                                                          │
│  docker/Dockerfile.edge                                  │
│  ├── ARM64 base image (L4T / JetPack)                    │
│  ├── ONNX Runtime GPU (CUDA provider)                    │
│  └── TensorRT Runtime (optional)                         │
│                                                          │
│  docker/docker-compose.prod.yml                          │
│  ├── PostgreSQL (production DB)                          │
│  ├── NGINX + TLS (reverse proxy)                         │
│  ├── Prometheus + Grafana (metrics)                      │
│  └── Resource limits per service                         │
└──────────────────────────────────────────────────────────┘
```

### CI/CD Pipeline

```
┌──────────────────────────────────────────────────────────┐
│               CI/CD PIPELINE [Phase 9]                    │
│                                                          │
│  .github/workflows/ci.yml                                │
│  ├── Lint (flake8 + black + isort)                       │
│  ├── Type Check (mypy)                                   │
│  ├── Test (pytest + coverage)                            │
│  ├── Build (Docker API + Dashboard)                      │
│  └── Model Accuracy Gate (mAP ≥ 0.78)                   │
│                                                          │
│  .github/workflows/deploy.yml                            │
│  ├── Multi-arch Docker build (amd64 + arm64)             │
│  ├── Push to GHCR                                        │
│  └── Tag with commit SHA + latest                        │
└──────────────────────────────────────────────────────────┘
```

### Updated Backend Architecture

```
api/
├── main.py                   ← FastAPI app, CORS, middleware, lifespan
├── routers/
│   ├── inspection.py         ← POST /inspections, POST /inspections/batch
│   ├── analytics.py          ← GET /analytics/summary, /defect-pareto, /drift
│   ├── reports.py            ← POST /reports/generate, GET /reports/{id}
│   ├── models.py             ← GET /models, PATCH /models/{id}/activate
│   └── websocket.py          ← WS /ws/live
├── middleware/
│   ├── auth.py               ← API key authentication
│   ├── audit_logger.py       ← Append-only JSONL audit trail
│   ├── metrics.py            ← Prometheus metrics collection     [Phase 7]
│   └── rate_limiter.py       ← Sliding window rate limiter       [Phase 7]
└── dependencies.py           ← DB session, model loader injection
```

---

*Architecture version: VisionFood QAI v2.0 (Production Build)*
