# VisionFood QAI

Intelligent Quality Inspection for Food & Beverage Manufacturing

VisionFood QAI is a full-stack inspection system that uses object detection, classification, and quality-scoring logic to automate visual inspection in food and beverage production lines. It includes model training and export utilities, a FastAPI backend, a React dashboard, and tooling for export to ONNX/TensorRT for edge deployment.

---

## TL;DR

- Purpose: Replace slow, inconsistent manual inspection with an automated vision pipeline that flags packaging and filling defects in real time.
- Components: detector (YOLOv11), classifier (EfficientViT), uncertainty estimation, REMEDY severity engine, FastAPI backend, React dashboard, ONNX export utilities.

---

## Features & Status

- Object detection pipeline (YOLOv8) — implemented (code), awaiting dataset for final training.
- Classification pipeline (EfficientViT) — implemented (code), awaiting dataset.
- Uncertainty quantification (MC Dropout) — implemented.
- REMEDY severity scoring & triage — implemented.
- FastAPI backend with middleware (auth, audit, metrics) — implemented.
- React dashboard (Vite + Tailwind) — implemented.
- Report generation (PDF) — implemented.
- ONNX export utilities — implemented; TensorRT planned.
- CI/CD, production hardening, explainability, drift detection — planned in later phases.

---

## Quick Links

- Code root: repository top-level
- Backend: `api/`
- Inference and preprocessors: `inference/`
- Training scripts: `training/`
- Export utilities: `export/`
- Dashboard: `dashboard/`
- Database and migrations: `database/`
- Models (artifacts, gitignored): `models/`
- Docker config: `docker/docker-compose.yml`, Dockerfiles in `docker/`

---

## Architecture Overview

The system follows a modular pipeline:

1. Capture: images from cameras or video streams (scripts in `scripts/` and `runs/`).
2. Inference: detection → crop → classify → uncertainty scoring (`inference/`).
3. REMEDY: severity scoring and triage (`remedy/`).
4. Backend: FastAPI serves inference, storage, reports, and metrics (`api/`).
5. Dashboard: React app shows live inspection feed, metrics, and reports (`dashboard/`).
6. Export/Edge: ONNX export and (planned) TensorRT for Jetson/ARM (`export/`, `docker/`).

Extended system and research-level architecture (multi-sensor fusion, federated learning, fog/edge orchestration) are part of the design but not required to run the core system.

---

## Repo Layout (short)

```
api/             # FastAPI server, routers, middleware
core/            # config, logging, shared utilities
inference/       # detector, classifier, pipelines
remedy/          # severity scoring, triage logic
dashboard/       # React app (Vite, Tailwind)
database/        # models, mongo models, migrations
training/        # training scripts and experiments
export/          # ONNX / (planned) TensorRT export
models/          # model artifacts (gitignored)
docker/          # Dockerfiles and compose files
scripts/         # helper scripts and extractors
tests/           # unit / integration / e2e tests
```

---

## Quick Start (Docker)

Prereqs: Docker, Docker Compose, Node.js (for dashboard development)

1. Start the stack (development):

```bash
cd docker
docker-compose up --build
```

2. Backend dev only:

```bash
python -m venv .venv
.venv\\Scripts\\activate   # Windows
pip install -r requirements.txt
alembic upgrade head
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

3. Dashboard dev:

```bash
cd dashboard
npm install
npm run dev
```

For full environment options and troubleshooting see the (removed) detailed run guide now consolidated below.

---

## Running Locally (summary)

- Configure environment variables: copy `.env.example` to `.env` and set `DATABASE_URL`, model paths, camera indexes, and any secrets.
- Database: migrations are managed with Alembic (`database/migrations/`). Run `alembic upgrade head` before starting backend.
- Start backend: `uvicorn api.main:app --reload --port 8000`.
- Start dashboard: use Vite dev server from `dashboard/`.
- Tests: run `pytest -q` from repo root. CI config is planned in `.github/`.

---

## Models & Data

- Dataset: expected under `data/` (gitignored). Typical layout: `data/raw`, `data/annotated`, `data/splits`.
- Trained artifacts: `models/` (not tracked in git). Use the scripts in `training/` to reproduce training.
- Export: ONNX export utilities in `export/` produce `models/*.onnx`. TensorRT conversion is planned.

---

## Training & Evaluation

- Training scripts live in `training/`. They use PyTorch and timm for model definitions.
- Experiment tracking can be used with MLflow or Weights & Biases as configured.
- Evaluation metrics: mAP for detection, Top-1 accuracy for classifiers, uncertainty calibration for UQ.

---

## Development Workflow

- Branching: feature branches off `main` or `develop`.
- Commits: small focused commits, reference issue/PR.
- Tests: add unit tests for new functionality and integration tests for API routes. Run `pytest` locally.
- Linting & formatting: use `ruff`/`black` if configured (check `pyproject.toml`).

---

## Team, Roles & Handoffs

- This repository contains role ownership and handoff checklists used during the project lifecycle. The previous per-role docs have been consolidated here.

---

## Project Status

- Core components implemented (inference code, backend, dashboard, report generation). Several production hardening items and final training runs remain.

---

## Where to Find More Details

Content from the original documentation (architecture, API reference, implementation plan, data model, run guide, prompt playbook, project status, and team guides) has been consolidated into this README. The repository previously contained individual Markdown reference files which have been merged to simplify onboarding.

If you need any of those original split documents restored, I can keep a backup branch or re-add them as separate files on request.

---

## Contributing

- Fork, create a feature branch, implement, add tests, and open a PR. Include a clear description and testing steps.

---

## License & Contact

Specify license here (e.g., MIT) and contact/maintainer details.

---

*Consolidated README generated on 2026-05-05.*
