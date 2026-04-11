# VisionFood QAI — Team Execution Guide

---

## Team Roles

| Role | Responsibilities | Phase Primary |
|------|-----------------|---------------|
| **ML Engineer** | Dataset, model training, ONNX export, UQ, benchmarking | Phases 0, 1, 2 |
| **Backend Engineer** | FastAPI, database, REMEDY engine, Celery, Redis | Phases 3, 4 |
| **Frontend Engineer** | React dashboard, WebSocket, charts, report UI | Phase 5 |
| **All** | Environment setup, integration testing, demo prep | Phases 0, 6 |

> For a 2-person team: ML Engineer covers Phases 0–2, Backend/Frontend Engineer covers Phases 3–5, integration in Phase 6 together.  
> For a solo project: follow phases sequentially as written.

---

## Git Workflow

### Branch Strategy

```
main          ← stable, demo-ready only. No direct commits.
dev           ← integration branch. Merge from feature branches.
feature/ph0-setup
feature/ph1-yolov11-training
feature/ph1-onnx-export
feature/ph2-efficientvit
feature/ph2-pipeline
feature/ph3-severity-scorer
feature/ph3-triage-router
feature/ph4-api-inspection
feature/ph4-api-analytics
feature/ph4-api-reports
feature/ph4-websocket
feature/ph5-live-feed
feature/ph5-analytics-page
feature/ph5-reports-page
feature/ph6-docker
feature/ph6-e2e-tests
```

### Commit Convention

```
<type>(<scope>): <short description>

Types: feat | fix | test | docs | refactor | chore

Examples:
  feat(inference): add ONNX runtime wrapper for YOLOv11
  fix(remedy): correct inverted PASS/FAIL logic in verdict router
  test(api): add integration tests for POST /inspect endpoint
  docs(ml): update ML explanations with focal loss rationale
  chore(deps): pin onnxruntime to 1.18.x
```

### Pull Request Process

1. Push feature branch: `git push origin feature/ph1-yolov11-training`
2. Open PR to `dev` on GitHub
3. PR title: `[Phase 1] YOLOv11 training script + ONNX export`
4. PR description must include:
   - What was implemented
   - Test results (mAP, latency, or test pass output)
   - Screenshot if frontend change
5. At least one team member must review and approve
6. Merge with **Squash and Merge** to keep `dev` history clean

### Merging `dev` → `main`

Only at:
- End of each phase (milestone merge)
- Demo day preparation

```bash
git checkout main
git merge dev --no-ff -m "chore: merge phase 1 complete → main"
git tag v0.1.0-phase1
git push origin main --tags
```

---

## Phase Execution Protocol

For every phase:

```
1. Pull latest dev
   git checkout dev && git pull origin dev

2. Create feature branch
   git checkout -b feature/ph<N>-<component>

3. Implement tasks from IMPLEMENTATION.md
   - Mark tasks as complete with [x] in IMPLEMENTATION.md as you go

4. Write/run tests
   pytest tests/unit/test_<module>.py -v

5. Open PR to dev
   - Paste test output in PR description

6. After PR merge, fill ROLE_HANDOFF_CHECKLIST.md for phase N

7. Announce phase complete in team channel
```

---

## Weekly Sync Agenda (30 minutes)

| Time | Item |
|------|------|
| 0–5 min | Quick status: what was done since last sync |
| 5–15 min | Blockers and risks — address before next week |
| 15–25 min | Demo latest working piece (model output, API, dashboard screenshot) |
| 25–30 min | Plan for next week — assign tasks from IMPLEMENTATION.md |

---

## Phase-by-Phase Execution Guide

### Phase 0 — Environment & Dataset (Weeks 1–2)

**All team members:**

```bash
# Each person sets up independently and verifies
git clone https://github.com/<your-org>/visionfood-qai.git
cd visionfood-qai
conda create -n visionfood python=3.11
conda activate visionfood
pip install -r requirements.txt
python -c "import torch, ultralytics, timm, fastapi; print('Setup OK')"
```

**ML Engineer:**
- Lead dataset collection and Roboflow annotation
- Share the Roboflow project URL with the team (read access for all)
- Export dataset to `data/annotated/` and commit `data/annotated/data.yaml` only (not images — add to `.gitignore`)

**Backend Engineer:**
- Set up Alembic: `alembic init database/migrations`
- Scaffold `core/config.py`, `core/schemas.py`
- Verify SQLite DB creation: `alembic upgrade head`

**Phase 0 ends when:**
- All team members: `pytest tests/unit/test_config.py` passes
- Dataset annotated, split verified, `data.yaml` in repo

---

### Phase 1 — YOLOv11 Training (Weeks 3–4)

**ML Engineer leads. Others review PR.**

Training is compute-heavy — run on:
1. **Google Colab** (free T4 GPU, free): Upload dataset, run `train_yolov11.py`
2. **Kaggle** (free T4/P100 GPU): Similar workflow
3. **Local GPU** if available

```python
# Colab / Kaggle training snippet
!pip install ultralytics wandb
import wandb
wandb.login()  # Paste API key from wandb.ai/settings

from ultralytics import YOLO
model = YOLO("yolov11n.pt")
model.train(data="/kaggle/input/visionfood/data.yaml", epochs=100, imgsz=640)
```

Download `best.pt` and `best.onnx` after training.  
Commit only ONNX to repo; PT file to shared drive (too large for git).

**Phase 1 ends when:**
- mAP@50 ≥ 0.80 on test set (paste confusion matrix in PR)
- `models/yolov11n_best.onnx` in repo
- `results/evaluation_report_phase1.md` written

---

### Phase 2 — Inference Pipeline (Week 5)

**ML Engineer implements. Backend Engineer reviews.**

Key files to implement:
```
inference/models/yolov11_detector.py
inference/models/efficientvit_classifier.py
inference/models/uq_inspector.py
inference/preprocessor.py
inference/postprocessor.py
inference/pipeline.py
inference/camera.py
```

Test pipeline with static images before camera:
```bash
python -m inference.run_single --image test_data/test_bottle.jpg
# Should output verdict, detections, UQ result
```

**Phase 2 ends when:**
- `tests/integration/test_pipeline_e2e.py` passes on 20 test images
- Pipeline gives correct verdict on ≥ 90% of test images

---

### Phase 3 — REMEDY Engine (Week 6)

**Backend Engineer leads. ML Engineer reviews severity weights.**

Files to implement:
```
remedy/severity_scorer.py
remedy/triage_router.py
remedy/sku_profile_manager.py
configs/sku_profiles/bottle_250ml.yaml
configs/sku_profiles/pouch_100g.yaml
configs/sku_profiles/can_330ml.yaml
```

Validate severity scoring manually:
```python
from remedy.severity_scorer import SeverityScorer
from inference.models.yolov11_detector import Detection

scorer = SeverityScorer()
det = Detection("label_misalignment", 0.9, (0.1, 0.1, 0.3, 0.4), 0.04)
result = scorer.score(det, attempt_count=0)
print(result)  # Should be S1 (label_misalignment + small area + high conf)
```

**Phase 3 ends when:**
- All `tests/unit/test_severity_scorer.py` and `test_triage_router.py` pass
- Pipeline + REMEDY integrated end-to-end

---

### Phase 4 — Backend & Database (Weeks 7–8)

**Backend Engineer leads.**

Implementation order within this phase:
1. Database models + migrations (4.1) — do first, everything depends on this
2. FastAPI app + dependencies (4.2)
3. Inspection router (4.3) — most critical endpoint
4. Analytics router (4.4)
5. WebSocket router (4.6) — needed before frontend
6. Reports router + PDF generator (4.5)
7. Model version router (4.7)

Test each endpoint immediately after implementation:
```bash
# Quick endpoint test with curl
curl -X POST http://localhost:8000/inspect \
  -H "X-API-Key: test_key" \
  -F "image=@test_data/test_bottle.jpg"
```

**Phase 4 ends when:**
- `pytest tests/integration/test_api.py` passes
- WebSocket receives events when pipeline runs
- PDF report generates successfully

---

### Phase 5 — Dashboard (Weeks 9–10)

**Frontend Engineer leads.**

Build pages in this order:
1. **LiveFeed** — highest priority for demo impact
2. **Analytics** — core of the quality report narrative
3. **History** — search and inspection record drill-down
4. **Reports** — PDF download
5. **Models** — model management (lowest priority)

Use `scripts/seed_db.py` to populate test data **before** building analytics:
```bash
python scripts/seed_db.py --records 500 --days 7
# Inserts 500 realistic inspections over last 7 days
```

Develop dashboard against seed data, not live camera, to avoid dependency on Phase 2.

**Phase 5 ends when:**
- All 5 pages render without errors
- Live feed shows bounding boxes via WebSocket
- Analytics charts load with real data from seeded DB

---

### Phase 6 — Integration & Demo (Weeks 11–12)

**All team members.**

#### Week 11: Integration
- Boot full stack with Docker Compose
- Run e2e test suite against full stack (not mocks)
- Fix integration bugs (CORS, auth, model path issues)
- Benchmark on demo hardware

#### Week 12: Demo Prep
- Physical products test: 20+ items covering all 4 defect classes
- Practice demo walkthrough (2–3 rehearsals minimum)
- Prepare fallback video (pre-recorded pipeline run)
- Final documentation review

**Demo Script:**
```
1. (2 min) Show slides: problem statement, architecture diagram
2. (3 min) Start system live: docker compose up
3. (5 min) Live inspection demo:
   - Scan clean product → PASS
   - Scan label misalignment → FAIL S1 → RELABEL action
   - Scan packaging damage → FAIL S2 → REPACK action
   - Scan contamination → FAIL S4 → REJECT
4. (2 min) Show dashboard: analytics, Pareto chart, history
5. (2 min) Download PDF quality report
6. (1 min) Q&A
```

---

## Code Quality Standards

```bash
# Lint Python (run before every commit)
flake8 . --max-line-length=100 --ignore=E203,W503

# Format Python
black . --line-length 100

# Lint + type check
mypy inference/ api/ remedy/ core/ --ignore-missing-imports

# JavaScript/React
cd dashboard && npm run lint
```

Add pre-commit hooks:
```bash
pip install pre-commit
pre-commit install
# .pre-commit-config.yaml (already in repo)
```

---

## Shared Resources

| Resource | Location |
|----------|---------|
| Roboflow dataset | `<share project URL in team chat>` |
| Trained model artefacts | Google Drive `visionfood-qai/models/` |
| W&B project | `wandb.ai/<team>/visionfood-qai` |
| MLflow tracking server | `http://localhost:5000` (local only) |
| GitHub repo | `https://github.com/<org>/visionfood-qai` |
