# VisionFood QAI — API Reference

---

## Base URL

```
Development:  http://localhost:8000
Production:   https://<your-domain>/api
```

Interactive docs available at: `http://localhost:8000/docs` (Swagger UI)  
Alternative: `http://localhost:8000/redoc`

---

## Authentication

All endpoints (except `/health` and `/ws/live`) require an API key:

```
Header: X-API-Key: <your-api-key>
```

Keys are configured in `.env`:
```
API_KEY=your_secret_key_here
```

---

## Endpoints

### Health

#### `GET /health`
Returns server health status. No authentication required.

**Response `200`**
```json
{
  "status": "ok",
  "model_loaded": true,
  "active_model": "yolov11n_v1.0.0",
  "uptime_seconds": 3600
}
```

---

### Inspection

#### `POST /inspect`
Submit a product image for defect inspection. Runs the full YOLOv11 → EfficientViT → UQ → REMEDY pipeline.

**Request** — `multipart/form-data`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `image` | File | Yes | JPEG or PNG, any resolution (resized internally to 640×640) |
| `product_id` | string | No | Product identifier (auto-generated UUID if omitted) |
| `sku` | string | No | SKU profile name (e.g. `bottle_250ml`). Defaults to `default` |

**Response `200`**
```json
{
  "product_id": "prod-20260321-001",
  "timestamp": "2026-03-21T10:30:00.123Z",
  "verdict": "FAIL",
  "overall_severity": "S2",
  "inference_ms": 42.7,
  "model_version": "yolov11n_v1.0.0",
  "escalated": false,
  "remedy_action": "RELABEL",
  "detections": [
    {
      "class_name": "label_misalignment",
      "confidence": 0.8731,
      "bbox": [0.12, 0.33, 0.48, 0.71],
      "area_fraction": 0.1296,
      "severity_grade": "S2",
      "remedy_action": "RELABEL",
      "uq_mean": 0.8731,
      "uq_std": 0.0421,
      "uq_ci_low": 0.7889,
      "uq_ci_high": 0.9573
    }
  ]
}
```

**Response `422`** — Image could not be decoded  
**Response `503`** — Model not loaded (startup in progress)

---

#### `GET /inspections`
Retrieve paginated inspection history with optional filters.

**Query Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `page` | int | 1 | Page number |
| `page_size` | int | 50 | Results per page (max 200) |
| `verdict` | string | — | Filter: `PASS`, `FAIL`, `ESCALATE`, `REVIEW` |
| `class_name` | string | — | Filter by defect class |
| `severity` | string | — | Filter: `S1`, `S2`, `S3`, `S4` |
| `date_from` | ISO datetime | — | Start of date range |
| `date_to` | ISO datetime | — | End of date range |

**Response `200`**
```json
{
  "total": 1250,
  "page": 1,
  "page_size": 50,
  "items": [
    {
      "id": "a3b4c5d6-...",
      "product_id": "prod-20260321-001",
      "timestamp": "2026-03-21T10:30:00Z",
      "verdict": "FAIL",
      "overall_severity": "S2",
      "inference_ms": 42.7,
      "defect_classes": ["label_misalignment"]
    }
  ]
}
```

---

#### `GET /inspections/{id}`
Retrieve full details of a single inspection.

**Path Parameter:** `id` — Inspection UUID

**Response `200`**
```json
{
  "id": "a3b4c5d6-...",
  "product_id": "prod-20260321-001",
  "timestamp": "2026-03-21T10:30:00Z",
  "verdict": "FAIL",
  "overall_severity": "S2",
  "inference_ms": 42.7,
  "model_version": "yolov11n_v1.0.0",
  "escalated": false,
  "operator_note": null,
  "detections": [ ... ],
  "remediation_action": {
    "action_type": "RELABEL",
    "station": "A",
    "outcome": "PASS",
    "created_at": "2026-03-21T10:30:01Z"
  }
}
```

**Response `404`** — Inspection not found

---

#### `PATCH /inspections/{id}/override`
Operator override — change a FAIL verdict to PASS with mandatory reason.

**Request Body**
```json
{
  "new_verdict": "PASS",
  "reason": "Defect below customer threshold for this batch"
}
```

**Response `200`** — Updated inspection  
**Response `403`** — Not authorised (operator role required)

---

### Analytics

#### `GET /analytics/summary`
Aggregated KPIs for a time period.

**Query Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `period` | string | `today` | `today`, `week`, `month`, or use `date_from`+`date_to` |
| `date_from` | ISO datetime | — | Custom range start |
| `date_to` | ISO datetime | — | Custom range end |

**Response `200`**
```json
{
  "total_inspected": 4820,
  "pass_count": 3921,
  "fail_count": 724,
  "pass_rate_pct": 81.35,
  "defect_rate_pct": 15.02,
  "remedy_save_rate_pct": 68.1,
  "avg_inference_ms": 44.2,
  "period_start": "2026-03-21T00:00:00Z",
  "period_end": "2026-03-21T23:59:59Z"
}
```

---

#### `GET /analytics/defect-rate`
Time-series defect rate, optionally broken down by class.

**Query Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `granularity` | string | `hourly` | `hourly`, `daily`, `weekly` |
| `class_name` | string | — | If provided, returns series for one class only |
| `date_from` | ISO datetime | — | Start |
| `date_to` | ISO datetime | — | End |

**Response `200`**
```json
{
  "series": [
    {"timestamp": "2026-03-21T10:00:00Z", "defect_rate": 0.142, "class_name": null},
    {"timestamp": "2026-03-21T11:00:00Z", "defect_rate": 0.189, "class_name": null}
  ]
}
```

---

#### `GET /analytics/pareto`
Defect class frequency for Pareto chart.

**Response `200`**
```json
{
  "items": [
    {"class_name": "label_misalignment",   "count": 312, "pct_of_total": 43.1, "cumulative_pct": 43.1},
    {"class_name": "packaging_damage",     "count": 198, "pct_of_total": 27.3, "cumulative_pct": 70.4},
    {"class_name": "improper_filling",     "count": 142, "pct_of_total": 19.6, "cumulative_pct": 90.0},
    {"class_name": "surface_contamination","count":  72, "pct_of_total": 9.9,  "cumulative_pct": 99.9}
  ]
}
```

---

#### `GET /analytics/severity-distribution`
Breakdown of defect severity grades.

**Response `200`**
```json
{
  "S1": 210,
  "S2": 312,
  "S3": 142,
  "S4": 60
}
```

---

### Reports

#### `POST /reports/generate`
Trigger asynchronous PDF report generation (runs as Celery task).

**Request Body**
```json
{
  "report_type": "shift",
  "period_start": "2026-03-21T08:00:00Z",
  "period_end": "2026-03-21T16:00:00Z",
  "generated_by": "operator_01"
}
```

**Response `202 Accepted`**
```json
{
  "id": "rpt-uuid-...",
  "status": "pending"
}
```

---

#### `GET /reports/{id}`
Check report generation status and get download URL.

**Response `200`**
```json
{
  "id": "rpt-uuid-...",
  "status": "complete",
  "pdf_url": "/reports/rpt-uuid-.../download",
  "generated_at": "2026-03-21T16:05:33Z"
}
```

Status values: `pending`, `complete`, `failed`

---

#### `GET /reports/{id}/download`
Download the generated PDF file.

**Response `200`** — `application/pdf` stream  
**Response `404`** — Report not found  
**Response `425`** — Report not yet generated (status = pending)

---

#### `GET /reports`
List all reports with status.

**Query Parameters:** `page`, `page_size`, `report_type`

---

### Model Management

#### `GET /models`
List all model versions with metadata.

**Response `200`**
```json
{
  "models": [
    {
      "id": "mv-uuid-...",
      "name": "yolov11n_v1.0.0",
      "architecture": "yolov11n",
      "stage": "production",
      "map50": 0.832,
      "map50_95": 0.741,
      "f1_score": 0.816,
      "latency_cpu_ms": 74.3,
      "latency_gpu_ms": 12.1,
      "is_active": true,
      "trained_at": "2026-03-10T00:00:00Z"
    }
  ]
}
```

---

#### `PUT /models/{id}/activate`
Promote a model version to production (requires auth).

**Response `200`** — Success with previous active model info  
**Response `404`** — Model version not found  
**Response `409`** — Model already active

---

#### `POST /models/{id}/rollback`
Rollback to the previous production model.

**Request Body**
```json
{
  "reason": "False negative spike detected on packaging_damage class"
}
```

**Response `200`** — Rollback successful  
**Response `400`** — No standby model available

---

### WebSocket

#### `WS /ws/live`
Real-time inspection event stream.

**Connection**
```javascript
const ws = new WebSocket("ws://localhost:8000/ws/live");
```

**Authentication** — pass API key as query param:
```
ws://localhost:8000/ws/live?api_key=<your-api-key>
```

**Message Format** (server → client, pushed on every inspection)
```json
{
  "event": "inspection",
  "data": {
    "product_id": "prod-20260321-001",
    "timestamp": "2026-03-21T10:30:00.123Z",
    "verdict": "FAIL",
    "overall_severity": "S2",
    "inference_ms": 42.7,
    "remedy_action": "RELABEL",
    "detections": [
      {
        "class_name": "label_misalignment",
        "confidence": 0.8731,
        "bbox": [0.12, 0.33, 0.48, 0.71]
      }
    ]
  }
}
```

**Heartbeat** — server sends every 30 seconds:
```json
{"event": "ping", "timestamp": "2026-03-21T10:30:30Z"}
```

Client should send `{"event": "pong"}` in response.

---

## Error Response Format

All error responses follow this structure:

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Image file is required",
    "details": {}
  }
}
```

| HTTP Status | Code | Description |
|------------|------|-------------|
| 400 | `BAD_REQUEST` | Invalid request format |
| 401 | `UNAUTHORIZED` | Missing or invalid API key |
| 404 | `NOT_FOUND` | Resource does not exist |
| 422 | `VALIDATION_ERROR` | Request body validation failed |
| 425 | `NOT_READY` | Resource not yet available (report generating) |
| 500 | `INTERNAL_ERROR` | Unexpected server error |
| 503 | `MODEL_NOT_LOADED` | Inference model not ready |

---

## Rate Limits

| Endpoint | Limit |
|----------|-------|
| `POST /inspect` | 120 requests/minute (matches target throughput of 120 products/min) |
| `GET /analytics/*` | 60 requests/minute |
| `POST /reports/generate` | 10 requests/minute |
| All other GET | 300 requests/minute |

Rate limit headers are included in all responses:
```
X-RateLimit-Limit: 120
X-RateLimit-Remaining: 87
X-RateLimit-Reset: 1742554260
```

---

## Phase 7–9 New Endpoints

### Batch Inspection [Phase 7]

#### `POST /inspections/batch`
Submit multiple product images in a single request for parallel inspection.

**Request Body**
```json
{
  "images": [
    {
      "image_b64": "<base64-encoded-image>",
      "product_id": "prod-001",
      "sku": "bottle_250ml"
    },
    {
      "image_b64": "<base64-encoded-image>",
      "product_id": "prod-002",
      "sku": "can_330ml"
    }
  ],
  "attempt_count": 0
}
```

**Response `201`**
```json
{
  "results": [ ... ],
  "summary": {
    "total": 8,
    "pass": 5,
    "fail": 2,
    "escalate": 1,
    "review": 0,
    "avg_latency_ms": 38.4
  }
}
```

Max batch size: 16 (configurable via `BATCH_MAX_SIZE`)

---

### Metrics [Phase 7]

#### `GET /metrics`
Prometheus metrics endpoint. No authentication required.

**Response `200`** — `text/plain; charset=utf-8`
```
# HELP visionfood_http_requests_total Total HTTP requests by method, path, and status code.
# TYPE visionfood_http_requests_total counter
visionfood_http_requests_total{method="POST",path="/inspections",status="201"} 1247

# HELP visionfood_inference_duration_seconds ML pipeline inference latency in seconds.
# TYPE visionfood_inference_duration_seconds histogram
visionfood_inference_duration_seconds_bucket{method="POST",path="/inspections",le="0.05"} 1100
visionfood_inference_duration_seconds_bucket{method="POST",path="/inspections",le="0.1"} 1200
visionfood_inference_duration_seconds_bucket{method="POST",path="/inspections",le="+Inf"} 1247
visionfood_inference_duration_seconds_sum{method="POST",path="/inspections"} 52.341
visionfood_inference_duration_seconds_count{method="POST",path="/inspections"} 1247

# HELP visionfood_model_loaded Whether the ML model is loaded (1) or not (0).
# TYPE visionfood_model_loaded gauge
visionfood_model_loaded{model="yolov11n"} 1
```

---

### Readiness Probe [Phase 7]

#### `GET /readiness`
Kubernetes-compatible readiness probe. No authentication required.

**Response `200`** — All systems healthy
```json
{
  "status": "ready",
  "checks": {
    "model_loaded": true,
    "database_reachable": true,
    "redis_reachable": true
  },
  "system": {
    "python_version": "3.11.9",
    "onnxruntime_version": "1.18.0",
    "cuda_available": false,
    "memory_rss_mb": 312.4,
    "uptime_seconds": 86400
  }
}
```

**Response `503`** — One or more checks failed
```json
{
  "status": "not_ready",
  "checks": {
    "model_loaded": false,
    "database_reachable": true,
    "redis_reachable": false
  }
}
```

---

### Drift Detection [Phase 8]

#### `GET /analytics/drift`
Returns model drift metrics and alert status.

**Response `200`**
```json
{
  "kl_divergence": 0.043,
  "alert": false,
  "threshold": 0.1,
  "window_size": 500,
  "baseline_distribution": {
    "improper_filling": 0.25,
    "packaging_damage": 0.25,
    "label_misalignment": 0.25,
    "surface_contamination": 0.25
  },
  "current_distribution": {
    "improper_filling": 0.28,
    "packaging_damage": 0.22,
    "label_misalignment": 0.27,
    "surface_contamination": 0.23
  },
  "last_updated": "2026-03-21T14:30:00Z"
}
```

---

### Explainability [Phase 8]

#### `POST /inspections` with `?explain=true`
When the `explain` query parameter is set, the response includes a Grad-CAM++ heatmap.

**Additional response fields:**
```json
{
  "...standard InspectionResult fields...",
  "explanation": {
    "heatmap_b64": "<base64-encoded-PNG>",
    "top_activation_region": {
      "x1": 0.15, "y1": 0.30, "x2": 0.45, "y2": 0.65
    },
    "method": "grad-cam++"
  }
}
```
