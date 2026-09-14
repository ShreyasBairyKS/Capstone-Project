# VisionQAI: End-to-End System Execution Guide

This guide provides step-by-step instructions for running the **VisionQAI Automated Bottle Cap Defect Detection Pipeline**. 

The system can be operated in two completely segregated execution modes:
1. **Mode 1: Native Terminal Commands** (Running directly on your Windows host workstation)
2. **Mode 2: Docker Containerized Orchestration** (Running in isolated containers with Docker Compose)

---

## Quick Navigation
- [Mode 1: Native Terminal Execution](#mode-1-native-terminal-execution-without-docker)
  - [1. Prerequisites](#1-prerequisites)
  - [2. Option A: Full-Stack via FastAPI (Recommended)](#option-a-full-stack-via-fastapi-recommended)
  - [3. Option B: Frontend Live Development (Hot Reloading)](#option-b-frontend-live-development-hot-reloading)
  - [4. Stopping the Native Server](#4-stopping-the-native-server)
- [Mode 2: Docker Containerized Execution](#mode-2-docker-containerized-orchestration)
  - [1. Prerequisites](#1-prerequisites-docker)
  - [2. Building and Starting Containers](#2-building-and-starting-containers)
  - [3. Accessing Containerized Endpoints](#3-accessing-containerized-endpoints)
  - [4. Live Code & Data Persistence Guarantees](#4-live-code--data-persistence-guarantees)
  - [5. Useful Docker Management Commands](#5-useful-docker-management-commands)
  - [6. Stopping the Docker Containers](#6-stopping-the-docker-containers)
- [Verification & Diagnostics](#verification--diagnostics)
- [Troubleshooting & FAQs](#troubleshooting--faqs)

---

# Mode 1: Native Terminal Execution (Without Docker)

Run this mode if you want direct host execution using your local Python virtual environment and Node.js toolchain.

### 1. Prerequisites
Ensure you have the required dependencies installed:
- **Python 3.11** virtual environment located at `.venv`
- **Node.js 18+ or 20+** (for frontend building)
- In PowerShell, ensure execution policy allows activation:
  ```powershell
  Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
  ```

---

### Option A: Full-Stack via FastAPI (Recommended)
In this mode, FastAPI serves both the **backend REST/WebSocket API** and the **compiled production React dashboard** on a single port (`8000`).

#### Step 1: Activate Virtual Environment
Open a PowerShell terminal at the repository root (`D:\Capstone Project code\Capstone-Project`):
```powershell
# Activate Python virtual environment
.\.venv\Scripts\Activate.ps1
```

#### Step 2: (One-Time) Build Frontend Production Assets
If you haven't built the frontend or made changes to React source files:
```powershell
cd pipeline/frontend
npm install
npm run build
cd ../..
```

#### Step 3: Launch the VisionQAI Unified Server
From the repository root:
```powershell
python -m uvicorn pipeline.api.app:app --host 127.0.0.1 --port 8000
```

#### What Happens on Startup:
1. **Zero-Touch Hardware Detection:** Detects NVIDIA CUDA GPU or falls back to multi-threaded CPU.
2. **Zero-Touch Auto-Compilation:** Automatically compiles `best.engine` (TensorRT FP16) on GPU or uses `best.onnx` on CPU.
3. **Warm-Up Passes:** Runs 2 dummy forward passes to preload weights into memory.
4. **Static Frontend Mount:** Mounts the compiled dashboard at `/visionqai`.

#### Access Points:
- **Operator Dashboard:** [http://localhost:8000/visionqai/](http://localhost:8000/visionqai/) *(or [http://localhost:8000/](http://localhost:8000/) which auto-redirects)*
- **API Health Check:** [http://localhost:8000/api/health](http://localhost:8000/api/health)
- **OpenAPI Swagger Docs:** [http://localhost:8000/visionqai/docs](http://localhost:8000/visionqai/docs)
- **Live WebSocket Stream:** `ws://localhost:8000/ws/stream`

---

### Option B: Frontend Live Development (Hot Reloading)
Use this option if you are actively editing React code in `pipeline/frontend/src/` and want instant Hot Module Replacement (HMR).

#### Terminal 1 (Backend Server):
```powershell
# From repository root
.\.venv\Scripts\Activate.ps1
python -m uvicorn pipeline.api.app:app --host 127.0.0.1 --port 8000 --reload
```

#### Terminal 2 (Vite Development Server):
Open a second PowerShell terminal:
```powershell
cd "D:\Capstone Project code\Capstone-Project\pipeline\frontend"
npm run dev
```
- Open your browser at: [http://localhost:5173/visionqai/](http://localhost:5173/visionqai/)
- Vite automatically proxies `/api` and `/ws` network requests to `http://127.0.0.1:8000`.

---

### 4. Stopping the Native Server

#### Standard Method:
Click into the terminal window running Uvicorn and press:
```
Ctrl + C
```

#### If Running in the Background or Port 8000 is Stuck:
Execute this one-liner in PowerShell to forcefully terminate whatever process is occupying port 8000:
```powershell
Get-NetTCPConnection -LocalPort 8000 | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force }
```

---

# Mode 2: Docker Containerized Orchestration

Run this mode to package and run the entire multi-tier system inside isolated containers (Single-PC Demo Mode).

```
+-----------------------------------------------------------------------------------+
| Host Workstation                                                                  |
|                                                                                   |
|  [ visionqai_backend ] (Port 8000)                                                |
|   - FastAPI REST API & WebSocket Server                                           |
|   - PyTorch / ONNX Runtime / TensorRT Acceleration                                 |
|   - Volume: ./pipeline (Live Code Sync)                                           |
|   - Volume: ./audit    (Database & Defect Storage)                                |
|   - Volume: weights/   (Preserves .pt, .onnx, .engine)                            |
|                                                                                   |
|  [ visionqai_frontend ] (Port 5173 & Port 80)                                     |
|   - High-performance Nginx reverse proxy                                          |
|   - Serves compiled React SPA                                                     |
|   - Proxies /api/ and /ws/ to visionqai_backend:8000                              |
+-----------------------------------------------------------------------------------+
```

### 1. Prerequisites (Docker)
1. Ensure **Docker Desktop** is installed and running on Windows.
2. Verify Docker CLI is accessible:
   ```powershell
   docker --version
   docker compose version
   ```
3. *(Optional for NVIDIA GPUs)* Ensure Docker Desktop has GPU support enabled under **Settings > Resources > WSL 2 / GPU**.

---

### 2. Building and Starting Containers

From the repository root (`D:\Capstone Project code\Capstone-Project`):

#### Option 1: Foreground Mode (View Live Server Logs in Console)
```powershell
docker compose up --build
```

#### Option 2: Detached Mode (Runs in Background)
```powershell
docker compose up -d --build
```

---

### 3. Accessing Containerized Endpoints

Once the containers are running:
- **VisionQAI Dashboard:** [http://localhost:5173/visionqai/](http://localhost:5173/visionqai/) or [http://localhost:80/visionqai/](http://localhost:80/visionqai/)
- **Backend API Direct:** [http://localhost:8000/visionqai/docs](http://localhost:8000/visionqai/docs)
- **API Health Check:** [http://localhost:8000/api/health](http://localhost:8000/api/health)

---

### 4. Live Code & Data Persistence Guarantees

The `docker-compose.yml` configuration includes persistent host bind mounts:
1. **Live Code Bind-Mount (`./pipeline:/app/pipeline`):**
   Any changes you make to Python files in `pipeline/` on your host machine reflect inside the container **immediately without needing to rebuild**.
2. **Weights Persistence Volume:**
   The container mounts your weights directory. Any auto-compiled `best.engine` or `best.onnx` created by the container is saved directly to your host disk and reused on next boot.
3. **Audit History Persistence (`./audit:/app/audit`):**
   The SQLite database (`inspections.db`) and raw bottle defect bundles are stored on your host machine. Rebuilding or restarting containers causes **zero data loss**.

---

### 5. Useful Docker Management Commands

#### Check Running Containers:
```powershell
docker compose ps
```

#### View Live Real-Time Logs:
```powershell
# Follow logs for all services
docker compose logs -f

# Follow logs for backend only
docker compose logs -f backend

# Follow logs for frontend/nginx only
docker compose logs -f frontend
```

#### Restart a Specific Container:
```powershell
docker compose restart backend
```

---

### 6. Stopping the Docker Containers

#### Stop and Preserve Data:
```powershell
docker compose down
```

#### Stop and Remove All Associated Volumes (Clean Reset):
```powershell
docker compose down -v
```

---

# Verification & Diagnostics

To verify system health and test pipeline components regardless of mode:

### 1. Query the Health Endpoint
```powershell
curl http://127.0.0.1:8000/api/health
```
Expected response:
```json
{
  "status": "online",
  "service": "VisionQAI Industrial Inspection Pipeline",
  "hardware": "CPU Fallback (14 threads)", // or "NVIDIA GeForce RTX ... (TensorRT FP16)"
  "classes": ["damaged_cap", "good_cap", "misplaced_cap", "no_cap", "open_cap", "wet_cap"]
}
```

### 2. Run the Full Test Suite
From the repository root:
```powershell
.\.venv\Scripts\Activate.ps1

python pipeline/test_step1.py              # Inference engine & NMS isolation
python pipeline/test_step2.py              # Asymmetric Quality Gate rules
python pipeline/test_step3.py              # Camera stream & queue management
python pipeline/test_step4.py              # Visual overlay rendering
python pipeline/test_step5.py              # SQLite audit & active learning bundler
python pipeline/test_step6.py              # FastAPI REST & WebSocket streaming
python pipeline/test_step_docker_prep.py   # Dockerfiles & auto-compile validation
```

---

# Troubleshooting & FAQs

### Q: Port 8000 or 5173 is already in use
**A:** Run this command in PowerShell to identify and kill the process holding the port:
```powershell
# Release Port 8000
Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force }

# Release Port 5173
Get-NetTCPConnection -LocalPort 5173 -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force }
```

### Q: How do I test the inspection flow without physical cameras?
**A:** Use the built-in simulation tools in the web dashboard:
1. Click **"Simulate Bottle"** in the top header to run a synthetic multi-camera inspection.
2. Click **"Run Conveyor"** to start an automated continuous inspection stream.
3. Click **"Upload & Inspect"** to test any custom image file directly from your computer.

### Q: Does Docker rebuild everything every time I make a code change?
**A:** No. Because `./pipeline` is mounted as a live host volume, edits to your Python files take effect immediately. You only need to run `docker compose up --build` if you modify `pipeline/requirements.txt` or `package.json`.
