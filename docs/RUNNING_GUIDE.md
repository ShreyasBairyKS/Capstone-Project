# VisionQAI: End-to-End System Execution Guide

This guide provides step-by-step instructions for running the **VisionQAI Automated Bottle Cap Defect Detection Pipeline**.

The system can be operated in two completely segregated execution modes:
1. **Mode 1: Native Terminal Commands** (Running directly on your host workstation without containers)
2. **Mode 2: Docker Containerized Orchestration** (Running in isolated GPU/CPU containers with Docker Compose)

---

## Quick Navigation
- [Host GPU Prerequisites (Manual Setup Required Before Running Docker)](#-host-gpu-prerequisites-manual-setup-required-before-running-docker)
  - [What Docker Can and Cannot Handle Automatically](#what-docker-can-and-cannot-handle-automatically)
  - [Setup for Windows GPU Host](#1-setup-for-windows-gpu-host)
  - [Setup for Linux (Ubuntu/Debian) GPU Host](#2-setup-for-linux-ubuntudebian-gpu-host)
  - [Host Verification Command](#3-host-gpu-verification-command)
- [Mode 1: Native Terminal Execution (Without Docker)](#mode-1-native-terminal-execution-without-docker)
  - [1. Prerequisites](#1-prerequisites)
  - [2. Option A: Full-Stack via FastAPI (Recommended)](#option-a-full-stack-via-fastapi-recommended)
  - [3. Option B: Frontend Live Development (Hot Reloading)](#option-b-frontend-live-development-hot-reloading)
  - [4. Stopping the Native Server](#4-stopping-the-native-server)
- [Mode 2: Docker Containerized Orchestration](#mode-2-docker-containerized-orchestration)
  - [1. Building and Starting Containers](#1-building-and-starting-containers)
  - [2. Accessing Containerized Endpoints](#2-accessing-containerized-endpoints)
  - [3. Live Code & Data Persistence Guarantees](#3-live-code--data-persistence-guarantees)
  - [4. Useful Docker Management Commands](#4-useful-docker-management-commands)
  - [5. Stopping the Docker Containers](#5-stopping-the-docker-containers)
- [Verification & Diagnostics](#verification--diagnostics)
- [Troubleshooting & FAQs](#troubleshooting--faqs)

---

# ⚡ Host GPU Prerequisites (Manual Setup Required Before Running Docker)

> [!IMPORTANT]
> **Why this manual configuration is required:**  
> Docker containers share the host operating system's kernel. While Docker **can and does automatically install** CUDA 12.4, cuDNN, TensorRT 10.x, PyTorch, and all Python libraries inside the image, **Docker CANNOT install physical hardware drivers or kernel-level GPU bridges onto your host operating system**.  
> You must perform this quick one-time setup on the GPU-enabled host machine before launching the Docker container.

### What Docker Can and Cannot Handle Automatically

| Component | Can Docker Install It? | Who Must Provide It? | Notes |
| :--- | :---: | :---: | :--- |
| **Physical NVIDIA GPU** | ❌ No | Host Machine | GTX 10xx+, RTX 20/30/40 series, A-series, or Jetson |
| **NVIDIA GPU Host Driver** | ❌ No | **Host Machine (One-Time)** | Docker cannot install kernel-level hardware drivers |
| **NVIDIA Container Toolkit** | ❌ No | **Host Machine (One-Time)** | The bridge that routes GPU access into containers |
| **CUDA Toolkit (User-space)** | ✅ **Yes** | **Docker Image** | Packaged directly inside `pipeline/Dockerfile.backend` |
| **cuDNN Runtime** | ✅ **Yes** | **Docker Image** | Packaged directly inside `pipeline/Dockerfile.backend` |
| **TensorRT 10.x & Bindings** | ✅ **Yes** | **Docker Image** | Installed via pip wheels (`tensorrt-cu12`) in container |
| **CUDA PyTorch (`cu121`)** | ✅ **Yes** | **Docker Image** | Installed automatically inside container |
| **Zero-Touch Auto-Compilation** | ✅ **Yes** | **Pipeline Code** | Compiles `best.engine` on first boot on that GPU |

---

### 1. Setup for Windows GPU Host

If your target GPU machine runs **Windows 10/11**:

1. **Install the NVIDIA GPU Driver:**
   - Download and install the latest Game Ready or Studio Driver from [nvidia.com/drivers](https://www.nvidia.com/Download/index.aspx).
2. **Install Docker Desktop with WSL 2:**
   - Download and install Docker Desktop.
   - During installation, verify that the **WSL 2 backend** option is checked.
   - In Docker Desktop, open **Settings > General** and verify **"Use the WSL 2 based engine"** is enabled.
3. *Note on Windows:* You do **not** need to install CUDA Toolkit or NVIDIA Container Toolkit manually on Windows. Docker Desktop on WSL 2 automatically bridges the host NVIDIA GPU into Docker containers.

---

### 2. Setup for Linux (Ubuntu/Debian) GPU Host

If your target GPU machine runs **Linux (Ubuntu 20.04 / 22.04 / 24.04)**:

1. **Install NVIDIA Display Drivers:**
   ```bash
   sudo apt update
   sudo apt install -y nvidia-driver-535  # or latest: nvidia-driver-550
   sudo reboot
   ```
   *(Verify installation with `nvidia-smi` after reboot).*

2. **Install the NVIDIA Container Toolkit (One-Time Host Bridge):**
   ```bash
   # Add NVIDIA package signing key and repository
   curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
   curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
     sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
     sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

   # Install the toolkit
   sudo apt update
   sudo apt install -y nvidia-container-toolkit

   # Configure Docker daemon to recognize the NVIDIA runtime
   sudo nvidia-ctk runtime configure --runtime=docker
   sudo systemctl restart docker
   ```

---

### 3. Host GPU Verification Command

Before starting the VisionQAI containers, run this single command on the host terminal to verify that Docker can access the physical GPU:

```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

- **Expected Output:** You should see the standard `nvidia-smi` status table showing your GPU name, driver version, and CUDA version.
- Once this command succeeds, your host machine is 100% prepared. Docker will now handle everything else automatically.

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
| Host Workstation (GPU Machine)                                                    |
|                                                                                   |
|  [ visionqai_backend ] (Port 8000)                                                |
|   - Base: nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04                            |
|   - Runtime: CUDA 12.4 + cuDNN + TensorRT 10.x + PyTorch                          |
|   - Auto-compiles best.engine (TensorRT FP16) on startup                          |
|   - Volume: ./pipeline (Live Code Sync)                                           |
|   - Volume: ./audit    (Database & Defect Storage)                                |
|   - Volume: weights/   (Preserves .pt, .onnx, .engine across rebuilds)            |
|                                                                                   |
|  [ visionqai_frontend ] (Port 5173 & Port 80)                                     |
|   - High-performance Nginx reverse proxy                                          |
|   - Serves compiled React SPA                                                     |
|   - Proxies /api/ and /ws/ to visionqai_backend:8000                              |
+-----------------------------------------------------------------------------------+
```

### 1. Building and Starting Containers

From the repository root (`D:\Capstone Project code\Capstone-Project`):

#### Option 1: Foreground Mode (View Live Server & GPU Logs in Console)
```powershell
docker compose up --build
```

#### Option 2: Detached Mode (Runs in Background)
```powershell
docker compose up -d --build
```

#### What You Will See in the Container Logs on Boot:
```text
[INFO] [pipeline.inference]: Active hardware runtime: NVIDIA GeForce RTX ... (target: cuda:0)
[INFO] [pipeline.inference]: TensorRT engine not found for this GPU. Starting on-the-fly compilation...
[INFO] [pipeline.inference]: Auto-compiling TensorRT FP16: best.pt -> best.engine (640x640, FP16)
[INFO] [pipeline.inference]: TensorRT compilation complete: weights/best.engine
[INFO] [pipeline.inference]: Warm-up complete. Ready for real-time inspection.
```

---

### 2. Accessing Containerized Endpoints

Once the containers are running:
- **VisionQAI Dashboard:** [http://localhost:5173/visionqai/](http://localhost:5173/visionqai/) or [http://localhost:80/visionqai/](http://localhost:80/visionqai/)
- **Backend API Direct:** [http://localhost:8000/visionqai/docs](http://localhost:8000/visionqai/docs)
- **API Health Check:** [http://localhost:8000/api/health](http://localhost:8000/api/health)

---

### 3. Live Code & Data Persistence Guarantees

The `docker-compose.yml` configuration includes persistent host bind mounts:
1. **Live Code Bind-Mount (`./pipeline:/app/pipeline`):**
   Any changes you make to Python files in `pipeline/` on your host machine reflect inside the container **immediately without needing to rebuild**.
2. **Weights Persistence Volume:**
   The container mounts your weights directory. Any auto-compiled `best.engine` or `best.onnx` created by the container is saved directly to your host disk and reused on next boot.
3. **Audit History Persistence (`./audit:/app/audit`):**
   The SQLite database (`inspections.db`) and raw bottle defect bundles are stored on your host machine. Rebuilding or restarting containers causes **zero data loss**.

---

### 4. Useful Docker Management Commands

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

### 5. Stopping the Docker Containers

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
  "hardware": "NVIDIA GeForce RTX ... (TensorRT FP16)", // or "CPU Fallback (14 threads)"
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

### Q: "could not select device driver '' with capabilities: [[gpu]]"
**A:** This means Docker cannot communicate with your NVIDIA GPU.
- **On Windows:** Ensure Docker Desktop is running and **"Use the WSL 2 based engine"** is enabled under Settings > General.
- **On Linux:** Ensure `nvidia-container-toolkit` is installed and Docker was restarted:
  ```bash
  sudo nvidia-ctk runtime configure --runtime=docker
  sudo systemctl restart docker
  ```

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
