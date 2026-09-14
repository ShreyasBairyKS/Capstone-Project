# Bottle Cap Defect Detection Pipeline: Architecture & Breakdown

> **Document Status:** 100% Fully Built, Verified & Containerized (VisionQAI)  
> **Location:** `D:\Capstone Project code\Capstone-Project\PIPELINE_BREAKDOWN.md`  
> **Model Checkpoint:** `YOLOv11seg-X model\YOLO_TrainingScripts\student_yolo11m_distilled\weights\best.pt` (yolo11m-seg)  
> **Target Deployment Hardware:** NVIDIA GPU (12GB VRAM) + Intel Multi-Core CPU + 32GB System RAM  
> **Fallback Hardware:** Multi-threaded CPU execution (automatic fallback if GPU is unavailable)  
> **Inference Acceleration:** Exported ONNX -> Compiled TensorRT FP16 (.engine) on target deployment machine

---

## 1. High-Level Concept

The goal is to take our trained `yolo11m-seg` model and build a complete, production-grade inspection system around it.

To keep the system clean, testable, and maintainable, the pipeline is divided into distinct, decoupled stages. Each stage has a single job. We can add, remove, or modify any section based on your specific requirements.

---

## 2. Proposed Pipeline Breakdown

```
[ Input Stream ] 
       │
       ▼
[ Part 1: Inference Engine ] ──── (Raw boxes, masks, confidences, latency)
       │
       ▼
[ Part 2: Inspection Decision Logic ] ──── (PASS, REJECT, UNCERTAIN, defect codes)
       │
       ├───────────────────────────────┐
       ▼                               ▼
[ Part 4: Visual Overlay ]     [ Part 5: Logging & Audit ]
 (Masks, boxes, HUD banners)    (CSV history, defect crops, edge cases)
       │                               │
       └───────────────┬───────────────┘
                       ▼
         [ Part 6: API Service Layer ]
           (FastAPI REST + WebSockets)
                       │
                       ▼
       [ Part 7: Frontend / Dashboard ] (Next Phase)
```

---

### Part 1: Core Inference Engine & Multi-Camera Ingestion `[ Completed & Verified ]`
* **Status:** `[ Completed & Verified ]`
* **Purpose:** Handles model initialization, hardware runtime dispatch, multi-view image ingestion, and batched forward passes on the NVIDIA GPU (with CPU fallback).
* **Why it exists:** A single camera has a 180° blind spot; comprehensive cap inspection requires a multi-camera array (e.g., front + rear, or 3-camera 120° coverage). Part 1 abstracts multi-view image acquisition and deep learning acceleration, ensuring camera synchronization and GPU batching happen in a single, ultra-low-latency pipeline.
* **Dual Input Modes:**
  1. **Mode A (Single Image / Offline Testing):** Accepts a single image filepath or raw NumPy array for debugging, dataset verification, and edge case testing.
  2. **Mode B (Synchronized Multi-Camera Array):** Connects to an array of cameras (e.g., Camera 1 [Front], Camera 2 [Rear/Opposite Side]) to eliminate blind spots around the entire circumference and seal of the bottle cap.
* **Low-Latency Multi-Camera Acquisition & Inference Architecture:**
  - **Asynchronous Parallel Capture:** Each camera runs in a dedicated non-blocking thread with minimal buffer (`buffer_size=1`) or hardware sync trigger, acquiring frames simultaneously in parallel so multi-camera capture adds near-zero latency.
  - **Batched TensorRT Forward Pass (Zero-Latency Scaling):** Instead of processing camera frames sequentially ($N \times \text{latency}$), the engine stacks $N$ camera views into a single batched tensor `[N, 3, 640, 640]` and executes **one single TensorRT FP16 forward pass**. On an NVIDIA GPU (12GB VRAM), batched inference across 2–3 cameras takes virtually the same time as a single frame (~2–4ms total).
  - **Strictly Isolated Per-Image NMS (Zero Cross-Camera Contamination):**
    - Because each camera captures an independent physical view, Non-Maximum Suppression (NMS) **must run in complete isolation for each index along the batch dimension $N$**.
    - **Never feed batched raw outputs into a single flat NMS kernel without batch offsets:** If Camera 1 (front) and Camera 2 (rear) both see a cap at similar coordinates `[x1, y1, x2, y2]` in their respective sensor frames, a global flat NMS would mistakenly treat them as duplicates and suppress one of them (potentially dropping a critical defect).
    - **Coordinate Isolation:** Pixel coordinates are strictly local to each individual camera's optical sensor. Coordinates must **never** be merged, averaged, or compared across different cameras.
    - **Keyed Post-Processing:** Inference outputs remain strictly partitioned in a keyed dictionary (e.g., `results['camera_front']`, `results['camera_rear']`).
    - **Aggregation at Decision-Tier Only:** Multi-camera outputs are combined exclusively at the business logic / decision level in Part 2 (e.g., *"If Camera 1 OR Camera 2 flags a defect -> trigger pneumatic reject actuator"*), never at the bounding-box coordinate level.
* **Model Formats & Compilation Pipeline:**
  - **Master Weights (`best.pt`):** Kept strictly intact as the permanent source-of-truth PyTorch checkpoint. Never overwritten or modified.
  - **ONNX Export (`student_yolo11m.onnx`):** Standalone exported model with dynamic batching enabled (`batch=1` to `batch=N`).
  - **TensorRT FP16 Compilation (`student_yolo11m_fp16.engine`):** Compiled from `.onnx` directly on the target deployment machine where the model runs.
  - **Runtime Priority:** TensorRT FP16 (Primary on GPU) -> ONNX Runtime (CUDA) -> PyTorch CUDA (`device=0`) -> Multi-threaded CPU Fallback.
* **Balanced Detection Floor & False Defect Prevention:**
  - Raw detection floor set to `0.35`: Captures low-confidence edge cases so subtle flaws are never dropped into blind spots.
  - Decoupled detection vs. defect classification: A low confidence score on `good_cap` (e.g. 0.55 due to surface glare) is retained with its raw probability vector, preventing false defect alarms.
* **Restructured Output Data Schema (Multi-View Aware):**
  - **1. Global Aggregated Summary:**
    - `hardware_used`: Active backend and batch size (e.g., `"NVIDIA GPU (TensorRT FP16, Batch=2)"`)
    - `total_inference_latency_ms`: Total execution time for the entire multi-camera forward pass
    - `camera_count`: Number of synchronized views processed
  - **2. Per-Camera View Detections (`views: Dict[str, CameraResult]`):**
    - For each camera view (`"camera_front"`, `"camera_rear"`):
      - `camera_id`: Identifier/name of the camera
      - `boxes`: Detected bounding boxes `[x1, y1, x2, y2]`
      - `masks`: Polygon segmentation contour coordinates
      - `class_names`: Detected classes (`good_cap`, `damaged_cap`, etc.)
      - `confidences`: Raw confidence floats (0.0 to 1.0)
      - `raw_frame`: Original camera frame (for visualizer / audit logging)

---

### Part 2: Multi-View Inspection Decision Logic (The Quality Gate) `[ Completed & Verified ]`
* **Status:** `[ Completed & Verified ]`
* **Purpose:** Aggregates multi-camera detections and translates them into a single, authoritative manufacturing decision (`PASS`, `REJECT`, `UNCERTAIN`).
* **Why it exists:** In a multi-camera setup, individual cameras may see different sides of the cap. Part 2 consolidates these views under strict quality control policies so that no defect on any side of the cap escapes unflagged.
* **Core Aggregation Policy: "Defect-Dominant" / Worst-Case Rule:**
  - **REJECT (Defect Detected):** If **ANY** camera view detects a defect (`damaged_cap`, `misplaced_cap`, `open_cap`, `wet_cap`, `no_cap`) with confidence >= defect threshold, the **entire bottle cap is immediately marked as REJECT**.
  - **PASS (All Compliant):** A bottle cap is granted **PASS** if and only if **ALL** camera views confirm compliance (`good_cap` >= pass threshold, e.g. 0.75).
  - **UNCERTAIN (Borderline / Re-Inspection):** If no view detects a confirmed defect, but one view has a borderline score (e.g. 0.40 - 0.75), the verdict is **UNCERTAIN** (routed for re-inspection or saved for review, preventing good caps from being falsely discarded).
  - **INSPECTION_FAILED (Anomaly):** If a camera view sees a bottle neck but fails to detect any cap at all.
* **Aggregated Output Structure:**
  - `global_verdict`: `"PASS"` | `"REJECT"` | `"UNCERTAIN"` | `"INSPECTION_FAILED"`
  - `primary_defect_type`: Name of the defect that triggered rejection (if any)
  - `flagged_cameras`: List of camera IDs that triggered the defect (e.g., `["camera_rear"]`)
  - `decision_reason`: Detailed explanatory string for operators and audit logs
  - `per_view_verdicts`: Individual verdict breakdown per camera angle

---

### Part 3: Stream & Ingestion Manager (Multi-Camera Array) `[ Completed & Verified ]`
* **Status:** `[ Completed & Verified ]`
* **Purpose:** Handles the physical/network camera hardware interfaces and parallel frame synchronization without adding latency.
* **Responsibilities:**
  - Support multi-camera synchronization: Parallel threaded ingestion for N cameras (USB3, GenICam, RTSP, or test image pairs).
  - Synchronized frame bundling: Packages simultaneous camera frames into a time-aligned bundle `[frame_cam1, frame_cam2]` with a shared `inspection_id`.
  - Frame queue buffer management: Drops stale frames (`queue_size=1`) to prevent latency buildup on continuous conveyors.
  - Offline test mode: Automatically pairs and feeds test images (e.g., front/rear pairs from folder).
* **Input:** Camera device configurations (IDs, RTSP streams, or local test directory).
* **Output:** Time-synchronized multi-view frame bundle ready for Part 1 batched inference.

---

### Part 4: Visual Overlay & Annotation Engine `[ Completed & Verified ]`
* **Status:** `[ Completed & Verified ]`
* **Purpose:** Generates visual feedback with semi-transparent segmentation overlays, defect tags, and confidence percentages.
* **Why it exists:** Operators need immediate visual confirmation of why a bottle was flagged. Decoupling annotation from storage allows the system to save pristine raw images for future retraining while rendering annotations dynamically whenever requested.
* **Core Responsibilities:**
  - **Semi-Transparent Defect Mask Overlay:** Renders alpha-blended transparent colored masks directly over the bottle cap defect contours (e.g. Red for confirmed defect, Amber for borderline/uncertain, Green for compliant). The underlying cap texture and defect ridge remain visible beneath the transparent tint.
  - **Defect Tagging with Confidence:** Annotates bounding boxes and masks with the exact defect label and confidence percentage (e.g., `damaged_cap: 92.4%`, `misplaced_cap: 87.1%`).
  - **Dual Annotation Modes:**
    1. **Live Stream Overlay:** Real-time HUD overlay on live conveyor video streams with global status banners (`[PASS]`, `[REJECT: DAMAGED_CAP]`, `[UNCERTAIN]`), FPS, and processing latency.
    2. **Dynamic On-Demand Review Rendering:** Reconstructs visual annotations on-the-fly from saved raw images and coordinate logs when an operator selects a historical defect to inspect.
* **Input:** Raw frame (live or loaded from audit storage) + detection metadata (boxes, masks, class names, confidences).
* **Output:** Cleanly annotated BGR frame with transparent masks and confidence labels.

---

### Part 5: Logging, Defect Audit & Telemetry (Bottle-Bundled Architecture) `[ Completed & Verified ]`
* **Status:** `[ Completed & Verified ]`
* **Purpose:** Stores comprehensive inspection records, archives raw multi-camera images per bottle, and enables active learning retraining.
* **Why it exists:** Provides complete quality traceability. Storing clean, unannotated images ensures that every captured defect can be reused directly as training data, while coordinate logs provide ready-to-use ground truth labels.
* **Strict Bottle-Level Bundling & Isolation Architecture:**
  - **Unique Bottle Inspection ID:** Every inspection event generates a unique session ID (e.g., `BOTTLE_20260912_120405_0042`).
  - **Dedicated Bottle Bundle Folder:** All multi-camera views for that specific bottle are bundled into an isolated directory:
    ```
    audit/bottles/BOTTLE_20260912_120405_0042/
    ├── camera_front.jpg       <- Raw, clean, unannotated original
    ├── camera_rear.jpg        <- Raw, clean, unannotated original
    ├── inspection_data.json   <- Bounding boxes, polygon mask contours, confidences, verdict
    └── labels/
        ├── camera_front.txt   <- Auto-generated YOLO polygon label for active learning
        └── camera_rear.txt    <- Auto-generated YOLO polygon label for active learning
    ```
  - **Strict Anti-Cross-Mixing Guarantee:** All images, coordinates, and decisions are strictly bound to their parent bottle ID. Images from different bottles are never mixed or grouped together.
  - **Multi-View Inspection Browser:** Operators can select a flagged bottle and seamlessly toggle between `camera_front`, `camera_rear`, or any other angle of that exact bottle.
* **Raw Image Storage + On-Demand Dynamic Annotation:**
  - **Storage Principle:** Defect images are saved **completely unannotated (raw)** to preserve pristine image pixels.
  - **Active Learning Ready:** The saved raw images and their accompanying polygon coordinate logs serve directly as new training datasets for fine-tuning future model iterations.
  - **Review Replay:** When a user or quality control officer opens a bottle record, Part 4 reads the raw image along with `inspection_data.json` and dynamically draws the transparent masks, bounding boxes, and confidence tags on-the-fly.
* **Telemetry & Structured Database:**
  - Appends every inspection event to SQLite/CSV (timestamp, bottle ID, global verdict, defect category, camera view breakdown, latency).
  - Maintains real-time rolling counters: Total inspected, pass rate %, defect breakdown tallies.

---

### Part 6: Backend Service & API Layer (FastAPI & WebSockets) `[ Completed & Verified ]`
* **Status:** `[ Completed & Verified ]`
* **Purpose:** Exposes the entire inspection and audit pipeline over high-speed REST and WebSocket network interfaces, decoupling the computer vision engine from any user interface.
* **Finalized Tech Stack:**
  - **Framework:** **FastAPI** (`fastapi == 0.111.1`) - Asynchronous, high-throughput ASGI framework.
  - **Server:** **Uvicorn** (`uvicorn == 0.30.6`) - Production-grade ASGI server with async event loops.
  - **Live Streaming:** **WebSockets** (`websockets == 12.0`) - Low-latency binary frame and telemetry streaming.
  - **Data Schemas:** **Pydantic v2** (`pydantic == 2.8.2`) - Strict validation for inspection outputs and configurations.
  - **Local Storage:** **SQLite** (built-in) - Serverless embedded database (`audit/inspections.db`) for tracking bottle records.
* **Key Endpoints Defined:**
  - `POST /api/inspect/image`: Manual upload of a single image or multi-camera pair for offline inspection.
  - `GET /api/stats`: Real-time rolling KPI metrics (total bottles, pass rate %, defect breakdown, avg latency).
  - `GET /api/defects`: Returns paginated list of historical defect events (bottle ID, timestamp, defect type, camera IDs).
  - `GET /api/defects/{bottle_id}`: Retrieves the complete multi-camera raw image set and polygon coordinate logs for a specific bottle.
  - `POST /api/config`: Dynamically updates operating thresholds (`conf_pass`, `conf_defect`) without restarting the service.
  - `WebSocket /ws/stream`: Full-duplex live stream sending synchronized multi-camera frames + real-time inspection telemetry.

---

### Part 7: Operator Frontend & SaaS Dashboard `[ Completed & Verified ]`
* **Status:** `[ Completed & Verified ]`
* **Purpose:** The user-facing web dashboard for operators and quality managers to monitor live inspections, review flagged defects, and configure system sensitivity.
* **Finalized Tech Stack:**
  - **Core Framework:** **React 18 / 19** with **Vite** build tooling.
  - **Styling:** **Vanilla CSS with Design Tokens** - Dark-mode industrial aesthetic, glassmorphic metric cards, high-contrast neon status badges (Emerald Green for `[PASS]`, Crimson Red for `[REJECT]`, Amber for `[UNCERTAIN]`).
  - **Real-Time Display:** Dual/Triple HTML5 `<canvas>` elements for zero-lag WebSocket rendering of front/rear camera feeds.
  - **Dynamic Review Engine:** HTML5 Canvas 2D overlay engine that dynamically draws semi-transparent polygon masks and confidence tags over raw review images.
  - **KPI Charts:** **Chart.js** for real-time pass-rate gauges, defect distribution bar charts, and latency monitors.

---

## 3. Docker Containerization Architecture: Single-PC Demo vs. Distributed Production

To ensure clean isolation between deep learning runtimes (CUDA / TensorRT) and web services, the entire architecture is containerized using **Docker** and **Docker Compose**.

```
========================================================================================
                          PRIMARY FOCUS: SINGLE-PC DEMO ARCHITECTURE
             (Entire multi-tier system running locally on your high-spec workstation)
========================================================================================

+--------------------------------------------------------------------------------------+
| Your Local Workstation (Intel CPU + 32GB RAM + 12GB NVIDIA GPU)                      |
|                                                                                      |
|  [ Container 1: vision-inference-engine ] (Port 8001 / Internal Network)             |
|   - Base: nvidia/cuda:12.x-runtime                                                   |
|   - GPU Access: Direct via NVIDIA Container Toolkit (--gpus all)                     |
|   - Responsibilities: Camera ingestion, Batched TensorRT FP16 inference, Isolated NMS |
|   - Footprint: ~1.5 GB VRAM | ~1.0 GB RAM                                            |
|                                                                                      |
|  [ Container 2: vision-backend-api ] (Port 8000 -> localhost:8000)                   |
|   - Base: python:3.11-slim                                                           |
|   - Responsibilities: FastAPI, SQLite DB, Audit file manager, WebSocket broker      |
|   - Persistence: Mounted local volume (./audit_data:/app/audit)                      |
|   - Footprint: 0 GB VRAM | ~150 MB RAM                                               |
|                                                                                      |
|  [ Container 3: vision-frontend-dashboard ] (Port 3000 -> localhost:3000)            |
|   - Base: nginx:alpine (serving compiled React + Vite build)                         |
|   - Responsibilities: User dashboard, multi-camera canvas, dynamic defect replay     |
|   - Footprint: 0 GB VRAM | ~50 MB RAM                                                |
+--------------------------------------------------------------------------------------+
         │
         ▼
   User Browser: Opens http://localhost:3000 (Full SaaS Experience, 100% Offline)
```

### 3.1 Primary Focus: Single-PC Demo Mode
* **Why this is our immediate emphasis:**
  - **Self-Contained & Deterministic:** The entire pipeline—from camera input and GPU inference to the SaaS web dashboard—runs concurrently on your local machine using Docker Compose (`docker compose up --build`).
  - **Zero External Costs or Dependencies:** Requires no active cloud subscription (AWS/GCP), no domain setup, and no internet access.
  - **Zero Demo Risk:** Demos, evaluation reviews, and capstone presentations will never stutter or fail due to poor conference Wi-Fi or cloud connection drops.
  - **Hardware Utilization:** Uses less than 2 GB of VRAM and 1.5 GB of RAM, leaving over 80% of your machine's resources free.

### 3.2 Future Roadmap: Distributed Production Mode (Post-Demo Bridge)
Once the single-PC demo milestone is fully built and verified, transitioning to real-world cloud SaaS requires **zero code rewrites**:

```
+------------------------------------+        HTTPS / WSS        +-----------------------------------+
| FACTORY EDGE PC (Local Workstation)| ─────────────────────────> | CLOUD SAAS PLATFORM (AWS / GCP)   |
|                                    | (Only Defect Bundles &     |                                   |
| - container-inference (TensorRT)   |  Telemetry Sent to Cloud)  | - container-cloud-api (FastAPI)   |
| - container-edge-agent (Local Sync)|                            | - container-cloud-frontend (React)|
| - Physical Cameras & Reject Actuator|                           | - Managed PostgreSQL & S3 Storage |
+------------------------------------+                            +-----------------------------------+
```
* In production:
  - The **Inference Container** remains on the factory edge PC next to the cameras and pneumatic reject actuator for sub-10ms response times.
  - The **Backend & Frontend Containers** deploy to cloud servers (AWS/GCP/Azure).
  - The communication bridge simply points from `localhost:8000` to `api.yourcloudplatform.com` via an environment variable (`.env`).

---

## 4. Step-by-Step Execution Plan

Now that all architectural specifications are locked in, our step-by-step roadmap is:

1. **Step 1: Part 1 `[ Completed & Verified ]` — Core Inference Engine**
   - Implement standalone model loading, NVIDIA GPU detection, and multi-threaded CPU fallback.
   - Implement isolated per-image NMS and batched forward pass support.
   - Verify locally on test image pairs.
2. **Step 2: Part 2 `[ Completed & Verified ]` — Multi-View Inspection Decision Logic**
   - Implement the "Defect-Dominant" aggregation rule (any defect = REJECT; all good = PASS; borderline = UNCERTAIN).
   - Test against sample normal, defect, and borderline cases.
3. **Step 3: Part 3 `[ Completed & Verified ]` — Stream & Ingestion Manager**
   - Implement multi-camera input handling and test pair ingestion.
4. **Step 4: Part 4 `[ Completed & Verified ]` & 5 — Visualizer, Audit Logger & Raw Defect Bundler**
   - Implement transparent mask rendering, bottle-isolated folder bundling, and raw image storage.
5. **Step 5: Part 6 `[ Completed & Verified ]` — FastAPI Backend & WebSockets**
   - Wrap the pipeline into async REST and WebSocket endpoints.
6. **Step 6: Part 7 `[ Completed & Verified ]` — React + Vite Frontend Dashboard**
   - Build the operator dashboard with live camera canvas and dynamic review drawer.
7. **Step 7: Docker Compose Integration `[ Completed & Verified ]`**
   - Package all components into the unified Single-PC Demo stack (`docker-compose.yml`).
