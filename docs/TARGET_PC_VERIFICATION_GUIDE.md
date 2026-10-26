# VisionQAI: Target PC Verification & Stress Test Execution Guide

This guide details the procedure for pulling the VisionQAI automated bottle cap defect inspection system onto the target GPU computer, configuring placeholders, verifying TensorRT dynamic batched compilation, and executing the closed-loop stress test.

---

> **CAUTION - MODEL FILES ARE NOT IN GIT (manual copy required)**
> The repository deliberately ignores `/models/`, `*.onnx` and `*.engine` because of their size.
> After cloning on the target PC you must copy these by hand before starting the system:
> - `models/fill-level-detector/` (`bottle_detector_best.pt`, `best.pt` or `water_surface_best.pt`; optional `.onnx` files) for Station 2
> - the Station 1 cap weights at `YOLOv11seg-X model/YOLO_TrainingScripts/student_yolo11m_distilled/weights/best.pt` (or point `VISIONQAI_CAP_MODEL_PATH` at them)
>
> If a model is missing, Station 2 fails to load (`FileNotFoundError` on station switch) and Station 1 fails at startup.
> Compiled `.engine` files are GPU-specific: never copy them between PCs; let the target PC compile its own.
> Set `VISIONQAI_CAP_SIM_DIR` / `VISIONQAI_FILL_SIM_DIR` in `.env` to your test image folders (see section 2).

## 1. Quick Checklist for Target PC Setup

| Step | Action | Command / File |
| :---: | :--- | :--- |
| **1** | Pull or clone the latest code | `git pull origin <branch>` |
| **2** | Copy configuration template | `copy pipeline\.env.target.example .env` |
| **3** | Configure target directories & variables | Edit `.env` (see Table below) |
| **4** | Verify Docker NVIDIA GPU access | `docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi` |
| **5** | Launch Containerized System | Double-click `start_system.bat` (or `docker compose up -d`) |
| **6** | Execute Closed-Loop Stress Test | `docker exec -it visionqai_backend python pipeline/stress_test.py` |

---

## 2. Environment Variables & Placeholder Configuration

When you pull the repository onto the target PC, create a `.env` file at the repository root by copying `pipeline/.env.target.example`:

```powershell
copy pipeline\.env.target.example .env
```

Open `.env` in any text editor and adjust the following placeholders:

| Variable | Default Value | Target PC Description / Placeholder |
| :--- | :---: | :--- |
| `VISIONQAI_ENGINE_VERSION` | `v1` | Version tag for the compiled TensorRT engine (`best_v1.engine`). Legacy static engine was `v0`. |
| `VISIONQAI_FRONTEND_VERSION` | `v1` | Version tag for Docker frontend image (`visionqai_frontend:v1`). |
| `VISIONQAI_BACKEND_VERSION` | `v1` | Version tag for Docker backend image (`visionqai_backend:v1`). |
| `VISIONQAI_TRT_BATCH_SIZE` | `3` | Batch size for multi-camera synchronization (3 side cameras: $0^\circ, 120^\circ, 240^\circ$). |
| `VISIONQAI_TRT_MAX_BATCH` | `4` | Maximum dynamic batch profile headroom. |
| `VISIONQAI_FORCE_RECOMPILE` | `0` | Set to `1` if you want to force recompile `best.pt` $\rightarrow$ `best_v1.engine` on boot. |
| `VISIONQAI_CONVEYOR_BPM` | `300` | Target production speed in Bottles Per Minute (300 BPM = 5 bottles/second). |
| `VISIONQAI_TEST_DATA_DIR` | `test` | **Path to your test dataset folder**. e.g., `D:\TestImages` or `test`. |
| `VISIONQAI_CAMERA_FRONT_SOURCE` | `0` | Camera 1 (Side View $0^\circ$): USB device index (`0`), RTSP URL, or directory. |
| `VISIONQAI_CAMERA_REAR_SOURCE` | `1` | Camera 2 (Side View $120^\circ$): USB device index (`1`), RTSP URL, or directory. |
| `VISIONQAI_CAMERA_SIDE_SOURCE` | `2` | Camera 3 (Side View $240^\circ$): USB device index (`2`), RTSP URL, or directory. |

---

## 3. Test Dataset Delivery & Folder Layout

The stress testing suite supports two directory layouts for test image ingestion:

### Layout A: Structured Bottle Subfolders (Recommended)
Organize your dedicated test bottles in subdirectories under `test/`:
```text
test/
  bottle_001/
    image1.png    <- Side Camera 1 (0 deg)
    image2.png    <- Side Camera 2 (120 deg)
    image3.png    <- Side Camera 3 (240 deg)
  bottle_002/
    image1.png
    image2.png
    image3.png
  ...
```

### Layout B: Flat Image Directory
If images are stored flat in a single directory (e.g. `test/images/`), the test runner automatically groups sequential images into synchronized triples.

### Recommended Test Set Size & Scenarios:
To perform a statistically rigorous industrial qualification run:
* **Recommended Total Count**: **50 to 100 unique bottles** (150 to 300 total images).
* **Scenario Distribution**:
  * **50–60% (25–50 bottles) Compliant / Good Caps**: Fully seated, correctly threaded, clean caps under varying rotation angles.
  * **15% (8–15 bottles) Damaged / Cut / Slit Caps**: Cuts, dented crowns, broken tamper-evident bridges positioned across camera seams.
  * **10% (5–10 bottles) Misplaced / Tilted Caps**: Crooked caps ($5^\circ$ to $15^\circ$ tilt) or high seated caps.
  * **10% (5–10 bottles) Wet / Condensation Caps**: Water droplets and wet surface film.
  * **5% (3–5 bottles) Missing / Extreme Defects**: Missing cap or crushed finish.

---

## 4. Execution Mode 1: Docker Containerized Orchestration (Production Target)

### 1. Launch the System
Execute the one-click Windows launcher from the repository root:
```powershell
.\start_system.bat
```
*What happens:*
1. Checks Docker Desktop and NVIDIA GPU availability.
2. Checks if `visionqai_backend:v1` and `visionqai_frontend:v1` exist. If missing, builds them automatically.
3. Starts containers via `docker compose up -d`.
4. Polls `http://localhost:8000/api/health` until the backend is online and the batched TensorRT engine is loaded.
5. Launches your default web browser to: [http://localhost:80/visionqai/](http://localhost:80/visionqai/).

### 2. Run the Stress Test Inside Docker
Open a terminal and run the stress test runner inside the backend container:
```powershell
docker exec -it visionqai_backend python pipeline/stress_test.py --iterations 100 --speed 300
```
Or pointing to a specific mounted directory:
```powershell
docker exec -it visionqai_backend python pipeline/stress_test.py --test-dir /app/test_data --iterations 100
```

### 3. Check Live Backend Container Logs
```powershell
docker compose logs -f backend
```
Look for the confirmation line:
```text
[INFO] [pipeline.inference]: Found existing compiled TensorRT engine (v1): .../best_v1.engine
[INFO] [pipeline.inference]: Batched warmup validated: single-pass batch=3 operational.
```

### 4. Stopping the System
```powershell
.\stop_system.bat
```

---

## 5. Execution Mode 2: Native Standalone Execution (Without Docker)

You can run the entire system natively on Windows without Docker:

### 1. Launch the Standalone System
```powershell
.\start_standalone.bat
```
*What happens:*
1. Activates Python `.venv` (`.\.venv\Scripts\Activate.ps1`).
2. Starts FastAPI serving both the backend REST/WebSocket API and the compiled React dashboard on port 8000.
3. Automatically opens [http://localhost:8000/visionqai/](http://localhost:8000/visionqai/).

### 2. Run the Stress Test Natively
```powershell
.\.venv\Scripts\python.exe pipeline/stress_test.py --test-dir test --iterations 50 --speed 300
```

---

## 6. Performance Benchmarks & Acceptance Criteria

When running the stress test on the target GPU machine, verify the output meets the following industrial criteria:

| Metric | Target PC GPU Standard (RTX 3060/4060+) | CPU Fallback Standard |
| :--- | :---: | :---: |
| **Batched Forward-Pass Latency ($p95$)** | **$\le 5.0\text{ ms}$** | $\le 45.0\text{ ms}$ |
| **Sequential Fallback Occurrences** | **0 (Zero)** | 0 (Zero) |
| **SQLite WAL Write Transaction ($p95$)** | **$\le 2.0\text{ ms}$** | $\le 5.0\text{ ms}$ |
| **DB Write / Lock Failures** | **0 (Zero)** | 0 (Zero) |
| **Effective Throughput** | **$\ge 300\text{ BPM}$ (5 bottles/sec)** | $\ge 60\text{ BPM}$ |
| **Actuator Decision Latency** | **$< 1.0\text{ ms}$** | $< 2.0\text{ ms}$ |
| **Memory Delta Across Run** | **$0\text{ MB}$ persistent growth** | $0\text{ MB}$ persistent growth |

---

## 7. Troubleshooting & FAQs

### Q: How do I force TensorRT to recompile if I change batch size or model?
**A:** In `.env`, set `VISIONQAI_FORCE_RECOMPILE=1` or increment `VISIONQAI_ENGINE_VERSION=v2`. On the next boot, the system will automatically compile `best_v2.engine` with the updated configuration.

### Q: How do I incorporate latest frontend UI updates into the Docker container?
**A:** In `.env`, increment `VISIONQAI_FRONTEND_VERSION=v2`, or run:
```powershell
.\start_system.bat --build
```
This forces a fresh multi-stage compilation of `pipeline/frontend/` into the Nginx container.

### Q: Where are stress test results saved?
**A:** Every run writes a full JSON report to `audit/stress_test_report.json`, containing all latency percentiles, defect distributions, and timestamped metrics.
