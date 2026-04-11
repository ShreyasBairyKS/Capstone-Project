# VisionFood QAI — Prompt Playbook

---

## Purpose

This playbook collects reusable GitHub Copilot / AI assistant prompts for every major development task in this project. Copy these prompts into Copilot Chat or your AI IDE assistant to get well-targeted, project-consistent code generation.

Each prompt is designed to produce output that follows the conventions in `ARCHITECTURE.md`, `DATA_MODEL.md`, and the existing codebase structure.

---

## Phase 0 — Setup & Dataset

### Generate `core/config.py`

```
I am building VisionFood QAI, a food defect detection system.
Generate `core/config.py` using Pydantic Settings (pydantic-settings package).
The class is EdgeConfig(BaseSettings).

Required fields with types and defaults:
- TIER: Literal["edge", "fog", "cloud"] = "edge"
- DEVICE_ID: str = "edge_node_01"
- YOLOV11_ONNX_PATH: Path = Path("models/yolov11n_best.onnx")
- EFFICIENTVIT_ONNX_PATH: Path = Path("models/efficientvit_m5_best.onnx")
- YOLOV11_CONF_THRESHOLD: float = 0.40
- YOLOV11_IOU_THRESHOLD: float = 0.45
- AUTO_PASS_THRESHOLD: float = 0.85
- ESCALATE_THRESHOLD: float = 0.60
- HUMAN_REVIEW_THRESHOLD: float = 0.45
- CAMERA_INDEX: int = 0
- CAMERA_MODE: Literal["software", "hardware"] = "software"
- REDIS_URL: str = "redis://localhost:6379"
- DATABASE_URL: str = Field(default=None, description="Set in .env file")
- LOG_LEVEL: str = "INFO"
- LOG_FORMAT: Literal["json", "text"] = "json"
- API_KEY: str = Field(default=None, description="Must be set in .env")
- REMEDY_ENABLED: bool = True

Config class should read from .env file.
Do not hardcode any passwords or secrets.
```

---

### Generate `scripts/visualise_annotations.py`

```
Generate a Python script `scripts/visualise_annotations.py` that:
1. Takes --data_dir argument pointing to data/annotated/train/
2. Randomly picks 5 images from data_dir/images/
3. Reads the corresponding .txt annotation from data_dir/labels/
4. Draws bounding boxes on the image using OpenCV
5. Shows class label text (from class list: improper_filling, packaging_damage, label_misalignment, surface_contamination)
6. Displays each image with cv2.imshow and waits for keypress
7. Saves annotated images to results/annotation_check/

Use argparse, pathlib.Path, numpy, opencv-python.
```

---

## Phase 1 — YOLOv11 Training

### Generate `training/train_yolov11.py`

```
Generate `training/train_yolov11.py` for VisionFood QAI.

Requirements:
- Use ultralytics YOLO class
- Load pretrained yolov11n.pt
- Read data path from config using EdgeConfig from core.config
- Initialize W&B run: wandb.init(project="visionfood-qai", name="yolov11n_v1")
- Train with: epochs=100, imgsz=640, batch=16, optimizer=AdamW, cos_lr=True
- Augmentations: mosaic=0.5, mixup=0.2, flipud=0.3, fliplr=0.5, degrees=15, hsv_s=0.3
- After training: log best mAP50, mAP50-95, precision, recall to W&B
- Copy best.pt to models/yolov11n_best.pt
- Export best.pt to ONNX with half=True, imgsz=640
- Copy best.onnx to models/yolov11n_best.onnx
- Log ONNX artefact to MLflow
- Print final summary table of per-class mAP
```

---

### Generate training evaluation script

```
Generate `training/evaluate_yolov11.py` that:
1. Takes --model (onnx path), --data (data.yaml path), --split (test) as arguments
2. Loads model with ONNX Runtime
3. Runs inference on all images in the test split
4. Computes: mAP@50, mAP@50-95, precision, recall, F1 per class, and overall
5. Generates confusion matrix using scikit-learn and saves to results/confusion_matrix.png
6. Generates PR curve matplotlib figure saved to results/PR_curve.png
7. Writes results/evaluation_report_phase1.md as a markdown table
8. Prints summary to stdout

Use: onnxruntime, numpy, opencv-python, scikit-learn, matplotlib, pathlib
```

---

## Phase 2 — Inference Pipeline

### Generate `inference/models/efficientvit_classifier.py`

```
Generate `inference/models/efficientvit_classifier.py` for VisionFood QAI.

The class EfficientViTClassifier wraps an ONNX Runtime session for EfficientViT-M5.

Requirements:
- __init__(self, onnx_path: str, device: str = "cpu")
- Input: 224×224×3 uint8 BGR numpy array
- preprocess(): resize to 224×224, BGR→RGB, normalise [0,1], HWC→NCHW, add batch dim
- classify(crop: np.ndarray) -> Tuple[int, str, float, np.ndarray]
  returns (class_id, class_name, top_confidence, all_class_probs)
- 4 class names: ["improper_filling", "packaging_damage", "label_misalignment", "surface_contamination"]
- Warmup with 5 dummy passes in __init__
- Use onnxruntime.InferenceSession with providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
- Include latency_ms measurement in return value
```

---

### Generate `inference/pipeline.py` verdict logic

```
I have inference/pipeline.py with class EdgeInferencePipeline.
The _apply_verdict_logic method currently has a bug — PASS and FAIL are inverted.

The correct logic should be:
  If there are NO detections at all: verdict = "PASS"
  
  If detections exist:
    If mean_confidence >= AUTO_PASS_THRESHOLD (0.85) AND NOT is_uncertain:
      verdict = "FAIL" (confirmed defect, high confidence)
      escalated = False
    
    Elif mean_confidence >= ESCALATE_THRESHOLD (0.60):
      verdict = "FAIL"
      escalated = True   (fog escalation flagged, but still fail)
    
    Elif mean_confidence >= HUMAN_REVIEW_THRESHOLD (0.45):
      verdict = "ESCALATE"
      escalated = True
    
    Else:
      verdict = "REVIEW"
      escalated = True

Rewrite _apply_verdict_logic(self, detections, uq, top) with correct logic.
Use threshold values from self.config (EdgeConfig).
```

---

### Generate camera capture loop

```
Generate `inference/capture_loop.py` for VisionFood QAI.

AsyncCaptureLoop class:
- __init__(self, pipeline: EdgeInferencePipeline, config: EdgeConfig)
- async run(self): infinite loop that captures frames and inspects them
- software_trigger mode: capture frame every 0.5 seconds (2fps default)
- hardware_trigger mode: read GPIO pin, fire on rising edge
- Use asyncio.Queue to pass frames to pipeline (non-blocking)
- On KeyboardInterrupt: graceful shutdown, print final latency stats
- Log every 10th result to stdout with: product_id, verdict, latency_ms, defect class if FAIL
- On ESCALATE/REVIEW verdict: log WARNING level
- Use OpenCV VideoCapture(config.CAMERA_INDEX)
```

---

## Phase 3 — REMEDY Engine

### Generate `remedy/triage_router.py`

```
Generate `remedy/triage_router.py` for VisionFood QAI.

class TriageRouter:
  Uses SeverityResult and Detection to determine remediation routing.
  
  @dataclass TriageDecision:
    action: str  # RELABEL | REFILL | REPACK | CLEAN | REJECT | PASS
    station: Optional[str]  # A | B | C | None
    is_remediable: bool
    reason: str
    max_attempts: int = 2
  
  def route(self, detection: Detection, severity: SeverityResult, attempt_count: int) -> TriageDecision:
    Full action map:
    - label_misalignment S1/S2 → RELABEL, Station A
    - improper_filling S1/S2 → REFILL, Station B
    - packaging_damage S1/S2 → REPACK, Station C
    - surface_contamination S1 → CLEAN, Station C
    - surface_contamination S2/S3/S4 → REJECT
    - anything S3/S4 → REJECT (no station)
    - attempt_count >= 2 → REJECT regardless of grade
    
  Reason string should be human-readable for audit log.
```

---

## Phase 4 — Backend

### Generate `api/routers/inspection.py`

```
Generate `api/routers/inspection.py` for VisionFood QAI FastAPI backend.

Requirements:
- APIRouter with prefix="/inspections", tags=["inspection"]
- POST /inspect endpoint:
  - Accepts UploadFile (image) + Form fields: product_id (optional), sku (optional, default="default")
  - Validates file is image (JPEG/PNG) — return 422 if not
  - Reads file bytes, decodes with cv2.imdecode to numpy array
  - Calls EdgeInferencePipeline.inspect(frame, product_id)
  - Persists InspectionResult to database via InspectionRepository
  - Returns InspectionResultSchema
  - Dependency injection: pipeline from get_pipeline(), db from get_db()
  
- GET /inspections endpoint:
  - Query params: page=1, page_size=50, verdict (Optional), class_name (Optional),
    severity (Optional), date_from (Optional[datetime]), date_to (Optional[datetime])
  - Returns paginated InspectionListItemSchema
  
- GET /inspections/{id}:
  - Returns full InspectionResultSchema including defects and remediation_action
  - 404 if not found

Use async def for all endpoint functions.
Import schemas from core.schemas.
```

---

### Generate `reports/generator.py`

```
Generate `reports/generator.py` for VisionFood QAI using ReportLab.

class QualityReportGenerator:
  def generate(self, report_data: dict, output_path: str) -> str:
    Generates a PDF quality report with:
    
    Page 1 — Cover:
      - Title: "VisionFood QAI — Quality Inspection Report"
      - Report type (Shift / Daily / Weekly)
      - Period: date_from to date_to
      - Generated at timestamp
      - Line separator
    
    Page 2 — Executive Summary:
      - Table: Total Inspected, Pass Count, Fail Count, Pass Rate %, Defect Rate %, REMEDY Save Rate %
      - Highlight defect rate in red if > 15%
    
    Page 3 — Defect Analysis:
      - Pareto bar chart (embed as image generated with matplotlib)
      - Severity distribution table (S1/S2/S3/S4 counts)
    
    Page 4 — Top Rejections:
      - Table of top 5 most severe rejections: product_id, class, severity, action
    
    Footer on every page: page number, report ID, "CONFIDENTIAL"
    
    report_data dict contains: all fields from QualityReport ORM model plus pareto_data list.
    Returns path to generated PDF file.
```

---

## Phase 5 — Dashboard

### Generate `BoundingBoxCanvas.jsx`

```
Generate React component `components/BoundingBoxCanvas.jsx` for VisionFood QAI dashboard.

Props:
  - imageBase64: string (base64 encoded JPEG from WebSocket message)
  - detections: Array of {class_name, confidence, bbox: [x1,y1,x2,y2], severity_grade}
  - width: number (canvas width)
  - height: number (canvas height)

Requirements:
- Use useRef for canvas element
- On imageBase64 or detections change: redraw canvas
- Draw image stretched to canvas dimensions
- For each detection:
  - Draw bounding box rectangle in class-specific colour:
    improper_filling = #3B82F6 (blue)
    packaging_damage = #F97316 (orange)
    label_misalignment = #8B5CF6 (purple)
    surface_contamination = #EF4444 (red)
  - Draw filled label background above box with class name + confidence (2dp)
  - Draw severity grade badge in top-right corner of box: S1=green, S2=yellow, S3=orange, S4=red
- If no detections, show "PASS ✓" text centered in green
- Handle canvas resize gracefully (do not distort bbox coordinates)
```

---

### Generate `hooks/useWebSocket.js`

```
Generate React hook `hooks/useWebSocket.js` for VisionFood QAI.

useWebSocket(url, onMessage):
  - Connects to WebSocket at url on mount
  - Calls onMessage(data) with parsed JSON on each "inspection" event
  - Implements exponential backoff reconnect:
    - First disconnect: retry after 1s
    - Second: 2s, third: 4s, max: 30s
  - Sends pong on receiving {"event": "ping"} heartbeat
  - Logs connection/disconnection with console.info
  - Cleans up (ws.close()) on unmount
  - Returns: { connected: boolean, lastEvent: object|null, reconnectCount: number }
  
Use useEffect, useRef, useCallback, useState.
Do not use any WebSocket library — native browser WebSocket only.
```

---

## Phase 6 — Testing & Docker

### Generate `tests/e2e/test_full_pipeline.py`

```
Generate `tests/e2e/test_full_pipeline.py` for VisionFood QAI.

Tests require full running stack (API + Redis + DB).
Use pytest with httpx.AsyncClient.

Test cases:
1. test_clean_product_pass: POST /inspect with clean_bottle.jpg → verdict == PASS
2. test_label_defect_fail: POST /inspect with label_skew.jpg → verdict == FAIL, class_name == "label_misalignment"
3. test_contamination_s4_reject: POST /inspect with contamination.jpg → severity_grade in [S3, S4], remedy_action == REJECT
4. test_inspection_persisted: POST /inspect → GET /inspections/{id} returns same product_id
5. test_analytics_updates: POST 5 FAIL inspections → GET /analytics/summary defect_rate_pct > 0
6. test_report_generation: POST /reports/generate → poll until status==complete → download PDF
7. test_websocket_event: POST /inspect while WS connected → WS receives event with matching product_id

Base URL: read from env var TEST_API_URL (default http://localhost:8000)
Test images: use files from tests/e2e/fixtures/ directory.
API key from env var TEST_API_KEY.
```

---

### Generate `docker/docker-compose.yml`

```
Generate `docker/docker-compose.yml` for VisionFood QAI.

Services:
1. redis:
   - Image: redis:7-alpine
   - Port: 6379:6379
   - Restart: unless-stopped
   - Healthcheck: redis-cli ping

2. api:
   - Build: docker/Dockerfile.api
   - Port: 8000:8000
   - Depends on: redis (healthy)
   - Volumes: ./models:/app/models (read-only), ./logs:/app/logs
   - env_file: .env
   - Healthcheck: curl http://localhost:8000/health
   - Restart: unless-stopped

3. celery_worker:
   - Build: docker/Dockerfile.api (same image as api)
   - Command: celery -A api.celery_app worker --loglevel=info -Q reports
   - Depends on: api, redis
   - Volumes: same as api
   - env_file: .env

4. dashboard:
   - Build: docker/Dockerfile.dashboard
   - Port: 3000:80
   - Depends on: api
   - Restart: unless-stopped

5. mlflow:
   - Image: ghcr.io/mlflow/mlflow:v2.14.0
   - Command: mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlflow.db
   - Port: 5000:5000
   - Volumes: mlflow_data:/mlruns

Networks: visionfood-net (all services on same network)
Volumes: mlflow_data
```

---

## General — Debugging Prompts

### Debug ONNX inference output

```
My YOLOv11 ONNX model returns raw output shape (1, 8400, 6).
I'm running it with onnxruntime and getting no detections even on images with obvious defects.

Confidence threshold is 0.40.
The 5th element (index 4) of each row should be the confidence score.
The 6th element (index 5) should be the class ID.

Can you write a debug function that:
1. Prints the max confidence value across all 8400 rows
2. Prints the top-5 highest confidence rows
3. Checks if values are in expected range [0, 1]
4. Checks if the ONNX model input expects BGR or RGB

And explain what might cause zero detections.
```

---

### Debug WebSocket not receiving events

```
My FastAPI WebSocket endpoint at /ws/live is connected but not receiving inspection events.
The POST /inspect endpoint works and returns correct results.

The flow: POST /inspect runs pipeline → publishes to Redis Stream "inspections:live" via XADD
          WebSocket router should consume "inspections:live" and push to clients.

Help me debug by:
1. Writing a test Redis consumer that reads from "inspections:live" directly
2. Checking the WebSocket router reads from the correct stream key
3. Identifying what could cause the stream to produce but the WS consumer to miss events
```

---

### Improve defect class recall

```
My YOLOv11 model has low recall on surface_contamination class (only 0.61).
The other 3 classes have recall >= 0.80.

Training dataset: 120 contamination images out of 520 total.
Most contamination images have small spot defects (< 5% of image area).

Suggest:
1. Which specific Albumentations augmentations would help small-spot detection
2. Whether I should adjust anchor sizes for this class
3. The mosaic augmentation parameter to increase
4. Whether switching to yolov11s (small) rather than yolov11n would significantly help
```
