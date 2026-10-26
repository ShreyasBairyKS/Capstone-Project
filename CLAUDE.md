# CLAUDE.md

VisionQAI: industrial bottle inspection system (college capstone). Two repo halves:

1. **Training code** (`YOLOv11seg-X model/`, `docs/YOLO11_Feature_KD_Guide.md`): finished. Do not modify.
2. **Inference + operator interface** (`pipeline/`): FastAPI backend + React dashboard that loads the trained models. This is the active work area.

The cap-classifier model in `models/cap-classifier/` duplicates the one under `YOLOv11seg-X model/.../student_yolo11m_distilled/weights/`. The interface currently uses the **weights/ copy**, not `models/cap-classifier/`.

## Stations

| | Station 1: Cap | Station 2: Fill Level |
|---|---|---|
| Code | `pipeline/{inference,decision,audit,visualizer,ingestion,api}` | `pipeline/fill_level/{inference,decision,audit,visualizer,api}` (mirrors Station 1's layout) |
| Model | YOLOv11m-seg student, 6 classes, 3 cameras batched | two YOLO detectors in `models/fill-level-detector/`: `bottle_detector_best.pt` + water meniscus (`best.pt` == `water_surface_best.pt`, identical files), 1 camera, batch=1 |
| REST | `/api/*` | `/api/fill/*` |
| WebSocket | `/ws/stream` | `/ws/fill/stream` |
| DB | `audit/inspections.db`, `audit/bottles/` | `audit/fill_level.db`, `audit/fill_level/bottles/` |
| Engine file | `best_<ver>.engine` next to `weights/best.pt` (dynamic batch) | `bottle_detector_<ver>.engine`, `water_surface_<ver>.engine` in `models/fill-level-detector/` (static batch=1) |

`fill_classifier_best.*` in `models/fill-level-detector/` is not used by the pipeline.

The stations are logically independent. They share one FastAPI app and one UI only for presentation convenience. Only one set of models should sit in GPU VRAM at a time:
- Cap engine is **eager-loaded at backend boot**. Fill engine is **lazy** (`FillPipelineService` starts in standby).
- UI station switch calls `POST /api/unload` → `POST /api/fill/load` (and the reverse). `StationProvider` also syncs GPU state on first page load.
- Load/unload is guarded by `_engine_lock` in both services. First GPU load compiles TensorRT, which takes minutes.

## Two runtime modes

1. **Standalone** (`start_standalone.bat`): native Windows `.venv`, `uvicorn pipeline.api.app:app` on `:8000`, serves the built frontend from `pipeline/frontend/dist` at `/visionqai/`. Hardware auto-detected: CUDA → compile/use TensorRT `.engine`; else `.onnx` (needs `onnxruntime`); else `.pt`. Loads `.env` from repo root.
2. **Docker** (`start_system.bat` → `docker compose` at **repo root**): CUDA 12.4 backend image + nginx frontend (`:80`, `:5173`). Compiles GPU-specific TensorRT FP16 engines. This is the mode the stress test runs against. `stop_system.bat` stops both.

Both compose files must stay in sync: root `docker-compose.yml` (used by `start_system.bat`) and `pipeline/docker-compose.yml` (context `..`). The root one is the one that runs.

## Commands

```bash
# Backend (from repo root; paths like audit/ are relative to CWD)
.venv\Scripts\python.exe -m uvicorn pipeline.api.app:app --host 127.0.0.1 --port 8000

# Frontend (pipeline/frontend; Vite base is /visionqai/, proxies /api and /ws to :8000)
npm run dev          # :5173
npm run build        # outputs dist/, mounted by app.py when present
npx oxlint src       # lint (many pre-existing warnings; only errors matter)

# Tests are plain scripts, not pytest. Run from repo root or any CWD with the venv python.
python pipeline/test_step1.py ... test_step6.py          # Station 1, one per layer
python pipeline/test_fill_step1.py ... test_fill_step6_frontend.py   # Station 2
python pipeline/test_system_e2e.py                       # both stations + VRAM swap, in-process
python pipeline/stress_test.py --iterations 50 --speed 300           # Station 1 latency/WAL stress
```

Tests write to `audit/` relative to CWD. Run from a scratch directory to avoid polluting repo data. They run in-process on CPU, so TensorRT paths are **not** covered by them.

## Architecture notes

- Pipeline per inspection: ingestion → `InferenceEngine` → `QualityGate` → `AuditLogger` (raw frames + YOLO labels + SQLite) → `VisualOverlayRenderer`. Overlays are rendered on demand; stored images stay clean for training.
- `PipelineService` / `FillPipelineService` are singletons (`get_instance()`) that own the engine, gate, logger, visualizer.
- Verdicts: `PASS`, `REJECT`, `UNCERTAIN` (borderline, goes to Active Learning), `INSPECTION_FAILED`. Cap gate: pass conf ≥ 0.75, defect conf ≥ 0.45, uncertain floor 0.40, defect-dominant across cameras.
- Fill ratio = (bottom − meniscus_y) / bottle height, after trimming 3% top / 2% bottom of the bottle box. Underfill < 0.40, overfill > 0.85, ±0.02 borderline band with low meniscus confidence → UNCERTAIN.
- Fill thresholds have **one source**: `VISIONQAI_FILL_UNDERFILL_THRESHOLD` / `VISIONQAI_FILL_OVERFILL_THRESHOLD` via `pipeline/fill_level/config.py`, read by gate, engine, overlay ruler, `/api/fill/health`. The frontend reads them from `fillHealth`.
- Frontend: `StationContext` holds the active station; `App.jsx` swaps WebSocket, endpoints, and page set (`components/` for cap, `components/fill_level/Fill*` for fill). Tabs: dashboard, audit, replay, annotation (active learning).
- **Paths come from `.env`** (loaded by `pipeline/env_loader.py`; real env vars win, e.g. Docker). Absolute paths work anywhere; relative paths resolve against the **repo root**, never CWD. Never hard-code a path in new code: use `env_path(NAME, default)`. Vars: `VISIONQAI_CAP_SIM_DIR`, `VISIONQAI_FILL_SIM_DIR` (images for Simulate), `VISIONQAI_CAP_MODEL_PATH`, `VISIONQAI_FILL_MODELS_DIR`, `VISIONQAI_AUDIT_DIR`. In Docker these are HOST paths that compose mounts to fixed container paths.
- Cap sim folder: subfolder per bottle (≤3 images) and/or flat images grouped by 3; each Simulate click advances to the next bottle. Fill sim folder: flat images, cycled.
- **Quality-gate settings are per station**: Station 1 `GET/POST /api/config` + `SettingsModal`; Station 2 `GET/POST /api/fill/config` + `FillSettingsModal` (runtime only, reset on restart; startup defaults come from `VISIONQAI_FILL_*`). `App.jsx` picks the modal by active station.
- Env vars: see `.env.example` and `pipeline/.env.target.example`. Engine versioning: `VISIONQAI_ENGINE_VERSION`, `VISIONQAI_FILL_ENGINE_VERSION`; force rebuild with `VISIONQAI_FORCE_RECOMPILE=1` / `VISIONQAI_FILL_FORCE_RECOMPILE=1`.
- All docs live in `docs/` (only this file stays at the root): `RUNNING_GUIDE.md`, `PROJECT_SETUP_GUIDE.md`, `PIPELINE_BREAKDOWN.md`, `TARGET_PC_VERIFICATION_GUIDE.md`, plus `IMPROVEMENTS_AND_EDGE_CASES.md` (analysis) and `TODO.md` (tracker for it; update the checkboxes as work lands).

## Gotchas

- `.gitignore` ignores `/models/`, `*.onnx`, `*.engine`, `/audit/`, `runs/`. The fill weights are **not in git**; copy `models/fill-level-detector/` to any new machine by hand. Docker bind-mounts it (compiled engines are cached there).
- If `VISIONQAI_CAP_SIM_DIR` is empty/missing, cap Simulate falls back to `runs/segment/predict*/`, then to flat gray frames (no real detections). Fill simulation defaults to `pipeline/fill_level/sample_images/`.
- Ultralytics TensorRT export writes an intermediate `.onnx` next to the `.pt` (it overwrites `bottle_detector_best.onnx`).
- Inference never runs on the event loop: async routes call services via `run_in_threadpool`. Services serialize inspections with `_infer_lock` (one TensorRT context is not thread-safe) and `unload_engine` waits for an in-flight inspection. Lock order is always `_infer_lock` → `_engine_lock`. Read `self.inference_engine` / `self._engine` once into a local before using it (a station switch can set it to None at any time).
- `.bat` files must stay CRLF; `.gitattributes` enforces it. Keep the model files manual-copy caution in `docs/TARGET_PC_VERIFICATION_GUIDE.md` current.
- Pydantic is v2: prefer `model_dump()`. Fill code still uses the deprecated `.dict()` (works, emits warnings).

## Status

- Station 1: verified on a CUDA machine (TensorRT).
- Station 2 + merged UI: verified in-process on CPU/ONNX only. **Not yet tested on GPU or in Docker.** First thing to check there: first-load TensorRT compile of both fill models (watch `docker compose logs -f backend`), then the swap Cap ↔ Fill, then VRAM release (`nvidia-smi`).
