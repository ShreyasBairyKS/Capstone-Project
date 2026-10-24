# VisionQAI: Cross-System Portability & Path Configuration Guide

This guide details **every single file and location** where paths, directories, and hardware settings are configured across the VisionQAI codebase. Use this checklist whenever cloning or transferring this project to a new target computer (e.g. from an RTX A5000 workstation to another desktop, laptop, or industrial edge PC) to prevent path mismatches, broken volumes, or missing dataset errors.

---

## 1. Quick Migration Checklist (The 3-Minute Setup)

When moving to a new computer, you only need to configure **two primary files**:

| Step | Action | Target File | What to Change |
| :---: | :--- | :--- | :--- |
| **1** | **Dataset & Hardware Config** | [`.env`](file:///.env) | Update `VISIONQAI_DATASET_DIR` to your new dataset folder path. |
| **2** | **Docker Volume Mount** *(if using Docker)* | [`docker-compose.yml`](file:///docker-compose.yml) | Update the host path on the left side of `/app/dataset:ro`. |
| **3** | **Check Model Weights** | [`YOLOv11seg-X model/.../weights`](file:///YOLOv11seg-X%20model/YOLO_TrainingScripts/student_yolo11m_distilled/weights) | Ensure `best.pt` exists (and optionally delete old `.engine` if GPU changed). |
| **4** | **Launch** | [`start_standalone.bat`](file:///start_standalone.bat) or [`start_system.bat`](file:///start_system.bat) | Run the launcher of your choice. |

---

## 2. Master Path Map: Where Every Path Lives

Here is the complete inventory of all files containing paths across the repository:

```
Capstone-Project/
├── .env                               <-- [PRIMARY] Global environment variables & dataset root
├── .env.example                       <-- Global template
├── docker-compose.yml                 <-- [PRIMARY for Docker] Volume mounts for host paths
├── start_standalone.bat               <-- Portable batch launcher (%~dp0 auto-resolves)
├── start_system.bat                   <-- Portable docker launcher (%~dp0 auto-resolves)
├── stop_system.bat                    <-- Portable cleanup script (%~dp0 auto-resolves)
├── pipeline/
│   ├── .env.target.example            <-- Pipeline-specific config template
│   ├── api/
│   │   ├── app.py                     <-- Relative mount of frontend /dist
│   │   └── pipeline_service.py        <-- Simulation image discovery search paths
│   ├── inference/
│   │   ├── engine.py                  <-- Model checkpoint & TensorRT engine path resolver
│   │   └── export.py                  <-- Model export directory resolver
│   ├── audit/
│   │   ├── db.py                      <-- SQLite WAL database path (`audit/inspections.db`)
│   │   └── logger.py                  <-- Audit images & bundle archive directory (`audit/`)
│   ├── stress_test.py                 <-- Test bundle discovery search paths
│   └── test_step1.py .. test_step6.py <-- Dynamic test image discovery routines
└── YOLOv11seg-X model/
    └── YOLO_TrainingScripts/
        ├── 01_merge_dataset.py        <-- Dataset root & merged export paths
        ├── 02_train_teacher.py        <-- data.yaml & runs output paths
        ├── 03_evaluate_teacher.py     <-- data.yaml & evaluation weights path
        ├── 04_generate_soft_labels.py <-- Dataset images & soft labels directory
        ├── 05_train_student_distill.py<-- data.yaml, teacher weights, student runs
        ├── 06_evaluate_and_compare.py <-- Evaluation comparison output paths
        ├── 07_export_student.py       <-- Export output paths (ONNX, TensorRT)
        └── student_yolo11m_distilled/
            └── weights/
                ├── best.pt            <-- Master PyTorch checkpoint (portable everywhere)
                └── best_v1.engine     <-- TensorRT engine (GPU & OS specific)
```

---

## 3. Detailed File-by-File Configuration

### File 1: `.env` (Repository Root) — **The Single Source of Truth**
This is the main configuration file used by both native mode and Docker.

```ini
# ==============================================================================
# 1. Dataset Paths (CHANGE THESE ON THE NEW COMPUTER)
# ==============================================================================
# Set this to the absolute folder where your dataset is located on the new computer.
# Examples:
#   Windows: VISIONQAI_DATASET_DIR="D:\Dataset\bottle_cap_sdp.v7i.yolov11"
#   Linux:   VISIONQAI_DATASET_DIR="/home/user/datasets/bottle_cap_sdp.v7i.yolov11"
VISIONQAI_DATASET_DIR="E:\P-25 Vision Food ai\dataset\bottle_cap_sdp.v7i.yolov11"

# Set this to the test subfolder (or leave pointing to /test inside the dataset)
VISIONQAI_TEST_DATA_DIR="E:\P-25 Vision Food ai\dataset\bottle_cap_sdp.v7i.yolov11\test"

# ==============================================================================
# 2. Camera Sources (Change if connecting physical industrial cameras)
# ==============================================================================
# 0, 1, 2 for USB / UVC cameras, or rtsp:// stream URL for IP cameras
VISIONQAI_CAMERA_FRONT_SOURCE=0
VISIONQAI_CAMERA_REAR_SOURCE=1
VISIONQAI_CAMERA_SIDE_SOURCE=2

# ==============================================================================
# 3. Model & Hardware Tuning
# ==============================================================================
VISIONQAI_ENGINE_VERSION=v1
VISIONQAI_TRT_BATCH_SIZE=3
VISIONQAI_TRT_MAX_BATCH=4
VISIONQAI_FORCE_RECOMPILE=0
```

> [!TIP]
> All Python scripts (`pipeline_service.py`, `stress_test.py`, test verification scripts, and YOLO training scripts) automatically read `VISIONQAI_DATASET_DIR` and `VISIONQAI_TEST_DATA_DIR` from `.env`. If you set it here, you won't need to touch any Python code!

---

### File 2: `docker-compose.yml` (For Docker Mode)
When running inside Docker, the Linux container cannot access Windows drive letters (`C:\`, `D:\`, `E:\`) unless they are mounted as volumes.

Open `docker-compose.yml` and check the **`volumes`** section under `backend`:

```yaml
    volumes:
      # 1. Live code bind-mount (Relative - no changes needed)
      - ./pipeline:/app/pipeline

      # 2. Model weights volume (Relative - no changes needed)
      - "./YOLOv11seg-X model/YOLO_TrainingScripts/student_yolo11m_distilled/weights:/app/YOLOv11seg-X model/YOLO_TrainingScripts/student_yolo11m_distilled/weights"

      # 3. Audit database & logs (Relative - no changes needed)
      - ./audit:/app/audit

      # 4. DATASET VOLUME: CHANGE THE LEFT SIDE TO YOUR NEW DATASET PATH
      # Syntax: "<Host Path on New PC>:/app/dataset:ro"
      # Examples:
      #   - "D:/MyDataset/bottle_cap_sdp.v7i.yolov11:/app/dataset:ro"
      #   - "C:/Projects/dataset:/app/dataset:ro"
      - "E:/P-25 Vision Food ai/dataset/bottle_cap_sdp.v7i.yolov11:/app/dataset:ro"
```

And in the **`environment`** section:
```yaml
    environment:
      # Inside the container, this always points to the container-internal mount:
      - VISIONQAI_DATASET_DIR=/app/dataset
      - VISIONQAI_TEST_DATA_DIR=/app/dataset/test
```

> [!IMPORTANT]
> In Docker on Windows, always use forward slashes `/` in `docker-compose.yml` volume paths (e.g. `D:/Dataset/bottle_cap` instead of `D:\Dataset\bottle_cap`) to prevent YAML escape syntax errors.

---

### File 3: Model Checkpoints & TensorRT Engines (`pipeline/inference/engine.py`)

#### How Paths are Resolved:
- Master Checkpoint: Resolves relatively from the code:
  `Path(__file__).resolve().parent.parent.parent / "YOLOv11seg-X model" / "YOLO_TrainingScripts" / "student_yolo11m_distilled" / "weights" / "best.pt"`
  **No manual edits needed.**

#### Crucial GPU Architecture Warning (`.engine` files):
- **TensorRT `.engine` files are compiled specifically for a single GPU model and operating system.**
- For example, `best_v1.engine` compiled on an **NVIDIA RTX A5000** will **NOT** load on an **RTX 3060, RTX 4090, or Tesla T4**, nor will a Windows-compiled `.engine` work on Linux.
- **What happens on a new PC:**
  - If you move to a PC with a different GPU, the engine loader will log a warning and **automatically fall back to `best.pt` using PyTorch CUDA** (zero crashes).
  - To recompile a fresh, native `.engine` optimized for the new GPU, simply set in `.env`:
    ```ini
    VISIONQAI_FORCE_RECOMPILE=1
    ```
    or delete the old `.engine` file in the `weights/` folder. The system will auto-compile a new engine on boot.

---

### File 4: Simulation Image Discovery (`pipeline/api/pipeline_service.py`)

When you click **SIMULATE** on the web dashboard, the backend feeds images to the 3 simulated camera angles.

In `pipeline/api/pipeline_service.py` -> `_find_sample_images()`:
```python
search_dirs = [
    # 1. Environment variables from .env (Recommended)
    Path(os.getenv("VISIONQAI_DATASET_DIR")),
    Path(os.getenv("VISIONQAI_TEST_DATA_DIR")),

    # 2. Hardcoded fallback (Update if not using .env)
    Path(r"E:\P-25 Vision Food ai\dataset\bottle_cap_sdp.v7i.yolov11"),

    # 3. Docker container internal path
    Path("/app/dataset"),

    # 4. Local repository test folder
    base_dir / "test",
]
```

**Rule for new PC:** If you define `VISIONQAI_DATASET_DIR` in `.env`, you do not need to modify this file.

---

### File 5: Stress Test Suite (`pipeline/stress_test.py`)

When running `python pipeline/stress_test.py`:
- Priority 1: `--test-dir <path>` argument if supplied via CLI.
- Priority 2: `VISIONQAI_TEST_DATA_DIR` or `VISIONQAI_DATASET_DIR` from `.env`.
- Priority 3: Fallbacks (`/app/dataset/test`, `/app/dataset`, local `test/` directory).

---

### File 6: YOLO Training & Distillation Pipeline (`YOLOv11seg-X model/YOLO_TrainingScripts/`)

If you plan to retrain or fine-tune the model on the new computer:
All 7 scripts use a portable relative root:
```python
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
```
And they discover the dataset via:
```python
DATASET_ROOT = Path(os.getenv("VISIONQAI_DATASET_DIR", r"E:\P-25 Vision Food ai\dataset\bottle_cap_sdp.v7i.yolov11"))
```

If your dataset is in a different location:
- Either set `VISIONQAI_DATASET_DIR` in `.env` or in your system environment.
- Or update the default string in:
  - [`01_merge_dataset.py`](file:///YOLOv11seg-X%20model/YOLO_TrainingScripts/01_merge_dataset.py#L25)
  - [`02_train_teacher.py`](file:///YOLOv11seg-X%20model/YOLO_TrainingScripts/02_train_teacher.py#L22)
  - [`03_evaluate_teacher.py`](file:///YOLOv11seg-X%20model/YOLO_TrainingScripts/03_evaluate_teacher.py#L20)
  - [`04_generate_soft_labels.py`](file:///YOLOv11seg-X%20model/YOLO_TrainingScripts/04_generate_soft_labels.py#L26)
  - [`05_train_student_distill.py`](file:///YOLOv11seg-X%20model/YOLO_TrainingScripts/05_train_student_distill.py#L25)
  - [`06_evaluate_and_compare.py`](file:///YOLOv11seg-X%20model/YOLO_TrainingScripts/06_evaluate_and_compare.py#L21)

---

### File 7: Frontend Web Dashboard (`pipeline/frontend/`)

- Source files are in `pipeline/frontend/`.
- Compiled production bundle is in `pipeline/frontend/dist/`.
- **How it connects to the backend:**
  - The React frontend uses relative API routes (e.g., `/api/health`, `/api/simulate`, `/ws/stream`).
  - It does **NOT** hardcode `http://localhost:8000` or IP addresses in production.
  - When served via FastAPI (`start_standalone.bat`) on port 8000 or via Nginx (`start_system.bat`) on port 80/5173, reverse proxy routing works automatically with zero path changes.

---

## 4. Portability Matrix Summary

| Component | Path Type | Hardcoded? | What to do on New Machine |
| :--- | :--- | :---: | :--- |
| **Dataset Images** | Absolute Host Path | Configurable via `.env` | **Edit `.env` (`VISIONQAI_DATASET_DIR`)** |
| **Docker Dataset Mount** | Host $\rightarrow$ Container | Set in `docker-compose.yml` | **Edit `docker-compose.yml` volume path** |
| **Model Weights (`best.pt`)** | Relative to repo | ❌ No | Fully automatic. Just ensure file exists. |
| **TensorRT Engine** | Relative to repo | ❌ No | Set `VISIONQAI_FORCE_RECOMPILE=1` if GPU changed. |
| **Audit Logs & DB** | Relative (`./audit/`) | ❌ No | Fully automatic. Creates directory automatically. |
| **Frontend Static SPA** | Relative (`pipeline/frontend/dist/`) | ❌ No | Fully automatic. Built once, runs anywhere. |
| **Batch Launchers** | Relative (`%~dp0`) | ❌ No | Automatically anchors to folder where bat is located. |

---

## 5. Step-by-Step New Machine Migration Procedure

Follow these steps when transferring this repository to a brand new computer:

### Step 1: Copy or Clone the Codebase
Extract or clone the repository to any drive or directory (e.g. `C:\VisionQAI` or `D:\Capstone-Project`).

### Step 2: Configure `.env`
1. Copy `.env.example` to `.env` if `.env` is not present.
2. Edit `.env` and update the dataset path to the folder where you placed the dataset on this new computer:
   ```ini
   VISIONQAI_DATASET_DIR="D:\Datasets\bottle_cap_sdp.v7i.yolov11"
   VISIONQAI_TEST_DATA_DIR="D:\Datasets\bottle_cap_sdp.v7i.yolov11\test"
   ```

### Step 3: Choose Your Running Mode

#### Option A: Running Standalone (No Docker)
1. Ensure Python 3.10+ is installed.
2. Create and activate a virtual environment:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\pip install -r pipeline\requirements.txt
   ```
3. Run the standalone launcher:
   ```cmd
   start_standalone.bat
   ```
4. Done! Browser will open to `http://localhost:8000/visionqai/`.

#### Option B: Running with Docker
1. Ensure Docker Desktop is installed and running with WSL2 backend.
2. Open `docker-compose.yml` and verify the dataset volume path matches your new dataset location:
   ```yaml
   - "D:/Datasets/bottle_cap_sdp.v7i.yolov11:/app/dataset:ro"
   ```
3. Run the docker launcher:
   ```cmd
   start_system.bat
   ```
4. Done! Browser will open to `http://localhost:80/visionqai/`.

---

## 6. Quick Verification Commands on the New Machine

After launching, run these quick sanity checks in terminal:

```powershell
# 1. Check API Health & GPU Model Status
curl http://127.0.0.1:8000/api/health

# 2. Trigger a 3-Camera Simulation Cycle
curl -X POST http://127.0.0.1:8000/api/simulate

# 3. Run Pipeline Unit Tests
python pipeline/test_step1.py
python pipeline/test_step2.py
python pipeline/test_step3.py

# 4. Run Industrial Conveyor Stress Test
python pipeline/stress_test.py --iterations 30
```
