# VisionFood QAI — Phase Handoff Checklists

---

## How To Use This Document

After completing each phase, the responsible team member runs through the checklist below. **All boxes must be checked before the next phase begins.** This prevents integration bugs caused by assumed-complete work.

Mark items complete with `[x]` and write the completion date beside each section header.

---

## Phase 0 → Phase 1 Handoff
> Completed by: __________ | Date: __________

### Environment
- [ ] All team members have `conda activate visionfood` working
- [ ] `python -c "import torch, ultralytics, timm, fastapi, onnxruntime"` succeeds for every team member
- [ ] `.env.example` committed to repo with all required keys documented
- [ ] `pytest tests/unit/test_config.py` passes

### Dataset
- [ ] Dataset annotated in Roboflow: minimum 400 real images (100/class)
- [ ] All 4 classes present: `improper_filling`, `packaging_damage`, `label_misalignment`, `surface_contamination`
- [ ] Dataset exported in YOLOv11 format and downloaded to `data/annotated/`
- [ ] `data/annotated/data.yaml` committed to repo
- [ ] Split verified: `train/`, `val/`, `test/` folders all non-empty
- [ ] `python scripts/visualise_annotations.py` shows correct labels (spot-checked 5 images)

### Codebase
- [ ] Folder structure matches `README.md` layout
- [ ] `core/config.py`, `core/schemas.py`, `core/logging.py` implemented
- [ ] `alembic upgrade head` creates tables without error
- [ ] `git log --oneline dev` shows commits with proper convention format

### Handoff Notes (Blockers / Decisions Made)
> _(write any decisions or caveats for Phase 1 ML Engineer here)_

---

## Phase 1 → Phase 2 Handoff
> Completed by: __________ | Date: __________

### Model Performance
- [ ] `models/yolov11n_best.onnx` exists and loads without error
- [ ] mAP@50 ≥ 0.80 on held-out test set (paste value: ______)
- [ ] No single class mAP < 0.70:
  - `improper_filling`:      ______
  - `packaging_damage`:      ______
  - `label_misalignment`:    ______
  - `surface_contamination`: ______
- [ ] Confusion matrix generated at `results/confusion_matrix.png`
- [ ] `results/evaluation_report_phase1.md` written with full metrics table

### Export
- [ ] ONNX export confirmed: `python -c "import onnxruntime; s=onnxruntime.InferenceSession('models/yolov11n_best.onnx'); print('OK')"`
- [ ] Latency benchmark run: CPU p95 < 80ms ✓ | GPU p95 < 15ms ✓ (circle applicable)
- [ ] W&B run ID logged: ______
- [ ] MLflow artefact committed

### Tests
- [ ] `pytest tests/unit/test_yolov11_detector.py` — all tests pass
- [ ] Test cases covered: blank image → no detection, synthetic defect → detection present, confidence threshold respected

### Handoff Notes
> _(Any classes with low recall? Specific failure modes observed? Notes for Phase 2 pipeline design)_

---

## Phase 2 → Phase 3 Handoff
> Completed by: __________ | Date: __________

### EfficientViT Classifier
- [ ] `models/efficientvit_m5_best.onnx` exists and loads
- [ ] Top-1 accuracy on crop test set ≥ 95% (value: ______)
- [ ] Focal loss used for training (confirmed in `training/train_efficientvit.py`)

### Inference Pipeline
- [ ] `inference/pipeline.py` — `EdgeInferencePipeline` class complete
- [ ] All verdict branches tested: PASS, FAIL, ESCALATE, REVIEW
- [ ] `inference/camera.py` — software trigger mode works (spacebar)
- [ ] Ring buffer implemented (32 frames)

### UQ Module
- [ ] `MCDropoutUQ` 20-pass inference working
- [ ] `uq_std` output is non-zero (dropout is actually active during inference — common bug)
- [ ] `UQResult` fields all populated: mean, std, ci_low, ci_high, is_uncertain, escalation_required

### Integration Test
- [ ] `pytest tests/integration/test_pipeline_e2e.py` passes
- [ ] Correct verdict on ≥ 90% of 20 test images

### End-to-End Pipeline Latency
- [ ] CPU: p95 latency ______ms (target < 150ms)
- [ ] GPU (if available): p95 latency ______ms

### Handoff Notes
> _(Confidence threshold values tuned? Any class performing poorly at UQ stage?)_

---

## Phase 3 → Phase 4 Handoff
> Completed by: __________ | Date: __________

### REMEDY Engine
- [ ] `SeverityScorer.score()` returns correct grade for all boundary cases
- [ ] All 16 `(class, grade)` → action mappings correct (see test cases below)
- [ ] Second-attempt penalty: S2 item pushed to S3 on second attempt (verified)
- [ ] `pytest tests/unit/test_severity_scorer.py` — all pass
- [ ] `pytest tests/unit/test_triage_router.py` — all pass

### Key Test Cases Verified
| Input | Expected Grade | Expected Action |
|-------|---------------|----------------|
| `label_misalignment`, conf=0.95, area=0.02, attempt=0 | S1 | RELABEL |
| `surface_contamination`, conf=0.85, area=0.15, attempt=0 | S3 | REJECT |
| `improper_filling`, conf=0.80, area=0.05, attempt=0 | S1 | REFILL |
| `packaging_damage`, conf=0.70, area=0.25, attempt=0 | S3 | REJECT |
| `label_misalignment`, conf=0.90, area=0.04, attempt=1 | S2 | RELABEL |

- [ ] All rows in table above verified: ______

### SKU Profiles
- [ ] `configs/sku_profiles/bottle_250ml.yaml` created
- [ ] `configs/sku_profiles/pouch_100g.yaml` created
- [ ] `configs/sku_profiles/can_330ml.yaml` created
- [ ] `sku_profile_manager.py` loads all 3 without error

### Pipeline Integration
- [ ] `InspectionResult` now includes `severity_grade`, `remedy_action`, `attempt_count`
- [ ] FAIL verdict correctly passes through REMEDY and appends action

### Handoff Notes
> _(Any severity weight adjustments made? Edge cases found?)_

---

## Phase 4 → Phase 5 Handoff
> Completed by: __________ | Date: __________

### Database
- [ ] All 5 tables created via Alembic: `inspections`, `defects`, `remediation_actions`, `model_versions`, `quality_reports`
- [ ] `alembic upgrade head` runs cleanly from a fresh checkout
- [ ] `scripts/seed_db.py` populated 500 records for FE development
- [ ] Active model version record exists in `model_versions` table

### API Endpoints
All endpoints tested with curl or Postman:
- [ ] `GET /health` → `{"status": "ok"}`
- [ ] `POST /inspect` with test image → `200` with correct `InspectionResultSchema`
- [ ] `GET /inspections` → paginated list
- [ ] `GET /inspections/{id}` → full detail
- [ ] `GET /analytics/summary` → KPIs populated from seeded data
- [ ] `GET /analytics/defect-rate` → time series data
- [ ] `GET /analytics/pareto` → 4 classes with counts
- [ ] `POST /reports/generate` → `202` with report ID
- [ ] `GET /reports/{id}` → status `complete` after Celery worker runs
- [ ] `GET /reports/{id}/download` → valid PDF download
- [ ] `GET /models` → list with at least 1 model version
- [ ] `WS /ws/live` → receives inspection events when `POST /inspect` is called

### Authentication
- [ ] Endpoints return `401` without `X-API-Key` header
- [ ] Correct key returns `200`

### Audit Logger
- [ ] `logs/audit.jsonl` created and growing on each request

### Integration Test
- [ ] `pytest tests/integration/test_api.py` — all pass

### Frontend Contract
- [ ] API response schemas match `core/schemas.py` — FE engineer has confirmed field names
- [ ] CORS configured for `http://localhost:3000`
- [ ] WebSocket endpoint tested and verified before FE starts building

### Handoff Notes
> _(Specific API schema decisions? Known slow queries? Celery gotchas?)_

---

## Phase 5 → Phase 6 Handoff
> Completed by: __________ | Date: __________

### Pages
- [ ] `LiveFeed` — WebSocket connects and renders bounding boxes on test images
- [ ] `Analytics` — all 4 charts render without errors using seeded data
- [ ] `History` — pagination, filters, row click → detail modal all working
- [ ] `Reports` — generate button triggers API, download works
- [ ] `Models` — model list rendered, activate/rollback buttons functional

### Components
- [ ] `BoundingBoxCanvas` — bbox coordinates correctly mapped from normalised [0–1] to pixel space
- [ ] `SeverityBadge` — S1=green, S2=yellow, S3=orange, S4=red
- [ ] `AlertPanel` — fires when 3+ FAILs in 30 seconds

### WebSocket
- [ ] Connection auto-reconnects on disconnect (tested by stopping API and restarting)
- [ ] Heartbeat pong implemented

### Cross-Browser
- [ ] Chrome ✓ | Firefox ✓ | Edge ✓ (tick applicable)

### Performance
- [ ] Analytics page loads in < 2 seconds with 500 seeded records
- [ ] Live feed renders new event within 200ms of WebSocket push

### Handoff Notes
> _(Known rendering issues? IE compatibility? Large dataset performance?)_

---

## Phase 6 — Final Checklist (Demo Day)
> Completed by: ALL | Date: __________

### Docker
- [ ] `docker compose up --build` starts without manual intervention
- [ ] All services healthy: API, Dashboard, Redis, Celery
- [ ] `http://localhost:8000/health` returns `model_loaded: true`
- [ ] `http://localhost:3000` loads dashboard

### E2E Tests
- [ ] `pytest tests/e2e/test_full_pipeline.py` — all pass
- [ ] Correct verdict on all 4 defect classes verified

### Performance
- [ ] Benchmark results recorded: p50=______ms, p95=______ms
- [ ] Throughput: ______products/min

### Demo Hardware
- [ ] Webcam working on demo laptop
- [ ] Physical products prepared: ≥ 5 per defect class
- [ ] Pre-recorded fallback video ready: `demo_fallback.mp4`
- [ ] Demo laptop fully charged, charger available

### Documentation
- [ ] `README.md` — quick start instructions verified working
- [ ] All phase sections of `IMPLEMENTATION.md` marked complete
- [ ] Final `results/evaluation_report_final.md` written

### Submission
- [ ] Code merged to `main`, tagged `v1.0.0-capstone`
- [ ] GitHub repo made public (or shared with evaluator)
- [ ] All model artefacts accessible via shared Google Drive link in README
- [ ] Recorded demo video uploaded and link added to README
