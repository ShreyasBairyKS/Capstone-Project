# YOLO11 Knowledge Distillation & Training Project Guide
**Target Architecture:** YOLO11x-seg (Teacher) -> YOLO11m-seg (Student)  
**Application Domain:** Automated High-Speed Bottle Cap Defect Detection & Segmentation  
**Repository Location:** `D:\Capstone Project code\Capstone-Project\`

---

## Table of Contents
1. [Conda Environment Setup Guide](#1-conda-environment-setup-guide)
   - [Prerequisites & CUDA Requirements](#prerequisites--cuda-requirements)
   - [Step-by-Step Environment Creation](#step-by-step-environment-creation)
   - [Package Installation](#package-installation)
   - [Dependency Specifications](#dependency-specifications)
   - [Environment Verification](#environment-verification)
2. [End-to-End Training Scripts Running Guide](#2-end-to-end-training-scripts-running-guide)
   - [Pipeline Overview & Execution Order](#pipeline-overview--execution-order)
   - [Script 01: Dataset Preparation & Merging](#script-01-dataset-preparation--merging)
   - [Script 02: Teacher Model Training (YOLO11x-seg)](#script-02-teacher-model-training-yolo11x-seg)
   - [Script 03: Teacher Model Evaluation](#script-03-teacher-model-evaluation)
   - [Script 04: Soft Labels Generation (Status & Note)](#script-04-soft-labels-generation-status--note)
   - [Script 05: Student Feature-Based Distillation (YOLO11m-seg)](#script-05-student-feature-based-distillation-yolo11m-seg)
   - [Script 06: Teacher vs. Student Evaluation & Comparison](#script-06-teacher-vs-student-evaluation--comparison)
   - [Script 07: Student Model Export & Optimization](#script-07-student-model-export--optimization)
3. [Comprehensive Path Reference & Directory Architecture](#3-comprehensive-path-reference--directory-architecture)
   - [Dual-Machine Architecture (Remote Server vs. Local Workstation)](#dual-machine-architecture)
   - [Master Path Reference Table](#master-path-reference-table)
   - [Cross-Machine Transfer & Path Portability Guide](#cross-machine-transfer--path-portability-guide)
4. [Model Storage, Weight Access & Ultralytics Evaluation Commands](#4-model-storage-weight-access--ultralytics-evaluation-commands)
   - [Saved Model Weights & Run Output Structure](#saved-model-weights--run-output-structure)
   - [How to Access and Load Trained Models in Python](#how-to-access-and-load-trained-models-in-python)
   - [Ultralytics CLI Commands (Val, Predict, Export, Benchmark)](#ultralytics-cli-commands)
   - [Python API Programmatic Evaluation](#python-api-programmatic-evaluation)
5. [Troubleshooting & Best Practices](#5-troubleshooting--best-practices)

---

## 1. Conda Environment Setup Guide

### Prerequisites & CUDA Requirements
- **Operating System:** Windows 10/11 or Linux (Ubuntu 20.04/22.04 LTS)
- **NVIDIA GPU Driver:** Driver version >= 525.60 (for CUDA 12.x) or >= 450.80 (for CUDA 11.8)
- **Package Manager:** Anaconda3 or Miniconda installed
- **Hardware Recommendations:**
  - **Teacher Training (YOLO11x-seg):** NVIDIA GPU with >= 16 GB VRAM (RTX 3090, RTX 4090, A5000, A100)
  - **Student Distillation (YOLO11m-seg):** NVIDIA GPU with >= 8 GB VRAM (RTX 3070, RTX 4070, or higher)

---

### Step-by-Step Environment Creation

#### 1. Open Terminal
Open **Anaconda Prompt** (Windows) or standard PowerShell / Bash with Conda initialized.

#### 2. Create Python 3.11 Conda Environment
```bash
conda create -n yolo_kd python=3.11 -y
conda activate yolo_kd
```

#### 3. Install PyTorch with CUDA Support
Choose the command matching your system's CUDA version:

- **For CUDA 12.1 / 12.4 (Recommended for RTX 30/40 Series):**
  ```bash
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
  ```
- **For CUDA 11.8 (Older GPU architectures):**
  ```bash
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
  ```
- **For CPU Only (Inference/Testing without GPU):**
  ```bash
  pip install torch torchvision torchaudio
  ```

---

### Package Installation

#### 4. Install Ultralytics and Vision Dependencies
YOLO11 requires `ultralytics>=8.3.0`.

```bash
# YOLO11 core framework
pip install "ultralytics>=8.3.0"

# Computer vision and image processing
pip install "opencv-python>=4.8.0" "pillow>=10.0.0" "albumentations>=1.3.0"

# Scientific computing and data handling
pip install "numpy>=1.24.0" "scipy>=1.11.0" "pandas>=2.0.0"

# Visualization and plotting
pip install "matplotlib>=3.7.0" "seaborn>=0.12.0"

# Utilities and progress tracking
pip install "tqdm>=4.65.0" "pyyaml>=6.0"
```

#### 5. Install Model Export and Acceleration Libraries
To enable exporting to ONNX, TensorRT, and running high-speed inference:

```bash
pip install "onnx>=1.14.0" "onnxsim>=0.4.33"
# For GPU-accelerated ONNX runtime:
pip install onnxruntime-gpu
# (Or CPU runtime if no GPU: pip install onnxruntime)
```

---

### Dependency Specifications

#### Option A: Single One-Line Installation
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install "ultralytics>=8.3.0" "opencv-python>=4.8.0" "pillow>=10.0.0" "numpy>=1.24.0" "scipy>=1.11.0" "pandas>=2.0.0" "matplotlib>=3.7.0" "seaborn>=0.12.0" "tqdm>=4.65.0" "pyyaml>=6.0" "onnx>=1.14.0" "onnxsim>=0.4.33" onnxruntime-gpu
```

#### Option B: Using requirements.txt
Save the following as `requirements.txt`:
```text
--extra-index-url https://download.pytorch.org/whl/cu121
torch>=2.1.0
torchvision>=0.16.0
torchaudio>=2.1.0
ultralytics>=8.3.0
opencv-python>=4.8.0
pillow>=10.0.0
numpy>=1.24.0
scipy>=1.11.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
tqdm>=4.65.0
pyyaml>=6.0
onnx>=1.14.0
onnxsim>=0.4.33
onnxruntime-gpu>=1.16.0
```
Install using:
```bash
pip install -r requirements.txt
```

---

### Environment Verification

Run this test command to confirm that GPU acceleration and all libraries are correctly recognized:

```bash
python -c "import torch, ultralytics; print('PyTorch Version :', torch.__version__); print('CUDA Available  :', torch.cuda.is_available()); print('Device Name     :', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'); print('Ultralytics Ver :', ultralytics.__version__)"
```

**Expected Console Output:**
```text
PyTorch Version : 2.x.x+cu121
CUDA Available  : True
Device Name     : NVIDIA GeForce RTX ... (or A100 / RTX 3090)
Ultralytics Ver : 8.3.x
```

---

## 2. End-to-End Training Scripts Running Guide

### Pipeline Overview & Execution Order

All training scripts reside in:  
`YOLOv11seg-X model/YOLO_TrainingScripts/`

| Step | Script | Purpose | Status in Current Setup | Execution Location |
| :--- | :--- | :--- | :--- | :--- |
| **01** | `01_merge_dataset.py` | Validates, sanitizes, and merges dataset splits | Completed | Local / Remote |
| **02** | `02_train_teacher.py` | Trains heavy `yolo11x-seg` teacher on dataset | Completed | Remote Training Server |
| **03** | `03_evaluate_teacher.py` | Evaluates teacher checkpoint & generates metrics | Completed | Remote / Local |
| **04** | `04_generate_soft_labels.py` | Precomputes offline soft-label logits | **SKIPPED** (Feature KD used) | Not Required |
| **05** | `05_train_student_distill.py` | Feature-based knowledge distillation for `yolo11m-seg` | **Active Training Script** | Local Workstation |
| **06** | `06_evaluate_and_compare.py` | Side-by-side benchmark (Teacher vs. Student) | Post-Distillation | Local Workstation |
| **07** | `07_export_student.py` | Exports student to ONNX / TensorRT / OpenVINO | Post-Evaluation | Local Workstation |

> **Crucial Architecture Note:**
> - **Teacher Model:** `yolo11x-seg` (Extra-large instance segmentation model, ~56.9M params, 319.6 GFLOPs)
> - **Student Model:** `yolo11m-seg` (Medium instance segmentation model, ~22.4M params, 110.6 GFLOPs)
> - **Distillation Strategy:** Feature-based distillation matching intermediate backbone representations ($P_3, P_4, P_5$) using $1\times1$ Conv projection adapters and Normalized MSE Loss.
> - **Script 04 is bypassed** because feature distillation operates dynamically online on feature activations during forward passes, rendering offline probability logit dumps obsolete.

---

### Script 01: Dataset Preparation & Merging
```bash
cd "D:\Capstone Project code\Capstone-Project\YOLOv11seg-X model\YOLO_TrainingScripts"

python 01_merge_dataset.py
```
- **Description:** Scans the dataset directory, resolves duplicate or misnamed folder splits (e.g., `lables` vs `labels`), verifies label-to-image associations, and structures data into clean `train/`, `valid/`, and `test/` splits.
- **Verification:** Ensure `data.yaml` is present and pointing to the respective split directories.

---

### Script 02: Teacher Model Training (YOLO11x-seg)
```bash
python 02_train_teacher.py
```
- **Description:**
  - Loads pretrained `yolo11x-seg.pt`.
  - Trains for 100 epochs with `imgsz=640`, `batch=16`, SGD optimizer, and CosineAnnealing learning rate schedule.
  - Applies focal loss (`fl_gamma=1.5`) to handle severe class imbalance between normal caps (`good_cap`) and rare defect classes (`scratch`, `dent`, `deformed`).
  - Uses Automatic Mixed Precision (AMP) for faster computation and lower VRAM usage.
- **Output:** Saves weights to `<PROJECT_DIR>/teacher_yolo11x_seg/weights/best.pt`.
- **Note:** This was run on the remote GPU server. Weights were subsequently transferred to the local project runs directory.

---

### Script 03: Teacher Model Evaluation
```bash
python 03_evaluate_teacher.py
```
- **Description:** Evaluates the teacher model against the validation and test splits. Computes:
  - Box mAP@50 and mAP@50-95
  - Mask mAP@50 and mAP@50-95
  - Per-class precision, recall, and F1 scores
- **Output:** Generates confusion matrices, PR curves, and batch prediction samples in the evaluation folder.

---

### Script 04: Soft Labels Generation (Status & Note)
> **STATUS: SKIPPED (Not Needed in Current Pipeline)**
- **Why it is skipped:** Script 04 was designed for response-based (logit) distillation where predicted class probabilities are dumped to disk. Because our system uses **Feature-Based KD** (distilling intermediate multi-scale feature maps inside the backbone), distillation runs completely on-the-fly. No offline soft-label files are required.

---

### Script 05: Student Feature-Based Distillation (YOLO11m-seg)
This is the primary script for training the distilled student model.

#### Pre-Flight Checklist:
1. Confirm `data.yaml` exists and path entries are valid.
2. Confirm teacher model weights exist at `TEACHER_WEIGHTS` (`best.pt` or `last.pt`).
3. Confirm GPU is available with >= 8 GB VRAM.

#### Execution Command:
```bash
python 05_train_student_distill.py
```

#### How the Feature KD Pipeline Works:
1. **Teacher Freezing:** The teacher model (`yolo11x-seg`) is loaded and its parameters are explicitly frozen (`param.requires_grad = False`, `teacher.eval()`). Teacher weights are never updated during training.
2. **Student Initialization:** The student model (`yolo11m-seg`) is initialized from pretrained weights with active gradients.
3. **Multi-Scale Feature Hooks:** Forward hooks intercept intermediate backbone feature maps at stages $P_3$, $P_4$, and $P_5$:
   - $P_3$ (Stride 8): Captures high-resolution micro-defect spatial details.
   - $P_4$ (Stride 16): Captures intermediate contextual features.
   - $P_5$ (Stride 32): Captures semantic defect classification features.
4. **Projection Adapters:** $1\times1$ convolutions align student channel dimensions to teacher channel dimensions:
   - $P_3$: 192 channels -> 384 channels
   - $P_4$: 384 channels -> 768 channels
   - $P_5$: 576 channels -> 1152 channels
5. **Loss Computation:**
   $$\mathcal{L}_{total} = \mathcal{L}_{YOLO}(\text{Student}, \text{GT}) + \alpha \cdot \mathcal{L}_{feat}(\text{Student}_{proj}, \text{Teacher})$$
   where $\alpha = 0.5$ balances task loss with feature distillation.
6. **Automatic Checkpoint Resumption:** If training is interrupted, running the script again automatically detects `last.pt` and resumes from the exact epoch where it stopped.

---

### Script 06: Teacher vs. Student Evaluation & Comparison
```bash
python 06_evaluate_and_compare.py
```
- **Description:** Evaluates both the teacher (`yolo11x-seg`) and distilled student (`yolo11m-seg`) on the exact same test dataset and hardware.
- **Metrics Compared:**
  - Detection Accuracy: Box mAP50, Box mAP50-95
  - Segmentation Accuracy: Mask mAP50, Mask mAP50-95
  - Efficiency: Parameter Count, GFLOPs, Model File Size (MB)
  - Speed: Latency per image (ms) and Throughput (FPS)
- **Output:** Formats and prints a clear markdown comparison table and summary report.

---

### Script 07: Student Model Export & Optimization
```bash
python 07_export_student.py
```
- **Description:**
  - Exports `student_yolo11m_distilled/weights/best.pt` to production formats:
    - **ONNX (FP32)** with dynamic or fixed batching.
    - **ONNX (FP16 Half-Precision)** for reduced memory footprint and 2x faster GPU inference.
    - Optional **TensorRT engine** generation if running on an NVIDIA GPU environment with TensorRT installed.
  - Automatically runs verification checks on the exported ONNX model using ONNX runtime.
- **Output Directory:** `YOLO_TrainingScripts/exported_models/`

---

## 3. Comprehensive Path Reference & Directory Architecture

### Dual-Machine Architecture

Because development is split between a remote server (for heavy teacher training) and a local workstation (for student distillation and git tracking), two filesystem paths exist:

```
+-----------------------------------------------------------------------------------+
| REMOTE TRAINING RIG (Drive E:)                                                    |
| - Role: Teacher training server                                                   |
| - Dataset: E:\Bottle_cap defect detection team 25\YOLOv11seg-X model\...        |
| - Runs:    E:\Bottle_cap defect detection team 25\...\runs\teacher_yolo11x_seg |
+-----------------------------------------------------------------------------------+
                                         |
                       Weights & Metrics Exported
                                         |
                                         v
+-----------------------------------------------------------------------------------+
| LOCAL WORKSTATION (Drive D:)                                                      |
| - Git Repository: D:\Capstone Project code\Capstone-Project\                    |
|   - Training Scripts: .\YOLOv11seg-X model\YOLO_TrainingScripts\                |
| - Dataset Directory: D:\Yolo Dataset\bottle_cap_sdp.v7i.yolov11\               |
| - Distillation Runs: D:\Yolo Dataset\YOLO_TrainingScripts\runs\...             |
+-----------------------------------------------------------------------------------+
```

---

### Master Path Reference Table

| Path Constant | Default Value in Code | Script(s) | Description / Role |
| :--- | :--- | :--- | :--- |
| `DATASET_DIR` | `D:\Yolo Dataset\bottle_cap_sdp.v7i.yolov11` | `01_merge_dataset.py` | Root directory containing raw unmerged dataset splits |
| `DATASET_YAML` (Remote) | `E:\Bottle_cap defect detection team 25\YOLOv11seg-X model\bottle_cap_sdp.v7i.yolov11\data.yaml` | `02`, `03` | Dataset configuration file on remote training server |
| `DATASET_YAML` (Local) | `D:\Yolo Dataset\bottle_cap_sdp.v7i.yolov11\data.yaml` | `05`, `06` | Dataset configuration file on local machine |
| `PROJECT_DIR` (Remote) | `E:\Bottle_cap defect detection team 25\YOLOv11seg-X model\YOLO_TrainingScripts\runs` | `02` | Output directory for teacher training on remote server |
| `PROJECT_DIR` (Local) | `D:\Yolo Dataset\YOLO_TrainingScripts\runs` | `05` | Output directory for student distillation runs |
| `RUN_NAME` (Teacher) | `teacher_yolo11x_seg` | `02`, `03`, `05`, `06` | Folder name for teacher run outputs |
| `RUN_NAME` (Student) | `student_yolo11m_distilled` | `05`, `06`, `07` | Folder name for student distillation run outputs |
| `TEACHER_WEIGHTS` | `D:\Yolo Dataset\YOLO_TrainingScripts\runs\teacher_yolo11x_seg\weights\best.pt` | `04`, `05`, `06` | Checkpoint of best trained teacher model |
| `TEACHER_FALLBACK`| `D:\Yolo Dataset\YOLO_TrainingScripts\runs\teacher_yolo11x_seg\weights\last.pt` | `05` | Fallback checkpoint if best.pt is unavailable |
| `STUDENT_WEIGHTS` | `D:\Yolo Dataset\YOLO_TrainingScripts\runs\student_yolo11m_distilled\weights\best.pt` | `06`, `07` | Checkpoint of best trained student model |
| `SOFT_LABELS_DIR` | `D:\Yolo Dataset\YOLO_TrainingScripts\soft_labels` | `04` | Folder for offline soft labels (legacy, unused in feature KD) |
| `EXPORT_DIR` | `D:\Yolo Dataset\YOLO_TrainingScripts\exported_models` | `07` | Destination folder for exported ONNX and TensorRT models |

---

### Cross-Machine Transfer & Path Portability Guide

When configuring the project on a new workstation or server:

1. **Verify or Edit `data.yaml`:**
   Open `bottle_cap_sdp.v7i.yolov11/data.yaml` and ensure the top-level `path:` points to your local dataset folder:
   ```yaml
   path: D:/Yolo Dataset/bottle_cap_sdp.v7i.yolov11  # Use forward slashes or escaped backslashes
   train: train/images
   val: valid/images
   test: test/images
   nc: 5
   names: ['good_cap', 'scratch', 'dent', 'deformed', 'crack']
   ```
2. **Placing Teacher Weights:**
   Ensure teacher weights are placed inside the project runs folder:
   `<PROJECT_DIR>/teacher_yolo11x_seg/weights/best.pt`
3. **Editing Script Paths:**
   In `05_train_student_distill.py`, ensure the constants at the top of the file match your current drive letters and folder paths.

---

## 4. Model Storage, Weight Access & Ultralytics Evaluation Commands

### Saved Model Weights & Run Output Structure

Every run creates a standardized results directory:

```text
runs/
+-- teacher_yolo11x_seg/
|   +-- weights/
|   |   +-- best.pt              <-- Top checkpoint (highest validation mAP)
|   |   +-- last.pt              <-- Latest epoch checkpoint (for auto-resume)
|   +-- args.yaml                <-- Snapshot of all hyperparameters
|   +-- results.csv              <-- Per-epoch training & validation losses and mAP
|   +-- confusion_matrix.png     <-- Validation confusion matrix
|   +-- confusion_matrix_normalized.png
|   +-- BoxF1_curve.png          <-- Bounding box F1 curve vs confidence
|   +-- BoxPR_curve.png          <-- Precision-Recall curve (Box)
|   +-- MaskF1_curve.png         <-- Segmentation mask F1 curve
|   +-- MaskPR_curve.png         <-- Precision-Recall curve (Mask)
|   +-- val_batch0_labels.jpg    <-- Ground-truth validation samples
|   +-- val_batch0_pred.jpg      <-- Model predicted detections & segmentations
|
+-- student_yolo11m_distilled/
|   +-- weights/
|   |   +-- best.pt              <-- Best student weights after distillation
|   |   +-- last.pt              <-- Last epoch student weights
|   +-- results.csv
|
+-- exported_models/
    +-- student_yolo11m_distilled.onnx        <-- Standard ONNX format
    +-- student_yolo11m_distilled_fp16.onnx   <-- Half-precision FP16 ONNX format
```

---

### How to Access and Load Trained Models in Python

#### 1. Basic Image Inference
```python
from ultralytics import YOLO

# 1. Load the trained distilled model weights
weights_path = r"D:\Yolo Dataset\YOLO_TrainingScriptsuns\student_yolo11m_distilled\weightsest.pt"
model = YOLO(weights_path)

# 2. Run prediction on an image
results = model.predict(
    source=r"D:\Yolo Datasetottle_cap_sdp.v7i.yolov11	est\images\sample.jpg",
    conf=0.25,        # Minimum confidence threshold
    iou=0.7,         # Non-Maximum Suppression (NMS) IoU threshold
    imgsz=640,
    device=0,        # GPU device 0 (use 'cpu' if running without GPU)
    save=True        # Save output annotated image
)

# 3. Process detection and segmentation results
for r in results:
    boxes = r.boxes
    masks = r.masks
    
    print(f"Total detections: {len(boxes)}")
    for box in boxes:
        cls_id = int(box.cls[0])
        class_name = model.names[cls_id]
        confidence = float(box.conf[0])
        coords = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
        print(f"  Detected: {class_name} | Confidence: {confidence:.2f} | BBox: {coords}")
        
    if masks is not None:
        polygons = masks.xy  # List of numpy arrays containing mask polygon coordinates
        print(f"  Extracted {len(polygons)} segmentation mask boundaries")
```

#### 2. Video Stream / Camera Inference
```python
from ultralytics import YOLO

model = YOLO(r"D:\Yolo Dataset\YOLO_TrainingScriptsuns\student_yolo11m_distilled\weightsest.pt")

# Run inference on a webcam feed (camera index 0)
model.predict(source=0, show=True, conf=0.3, imgsz=640)
```

---

### Ultralytics CLI Commands

The Ultralytics command-line interface provides fast, script-free access to model evaluation, prediction, and export:

#### 1. Model Validation (Calculate mAP, Precision, Recall on Test Set)
```bash
# Validate Distilled Student Model
yolo segment val   model="D:\Yolo Dataset\YOLO_TrainingScriptsuns\student_yolo11m_distilled\weightsest.pt"   data="D:\Yolo Datasetottle_cap_sdp.v7i.yolov11\data.yaml"   split=test   imgsz=640   batch=16   device=0   project="D:\Yolo Dataset\YOLO_TrainingScriptsuns"   name=val_student_test

# Validate Teacher Model
yolo segment val   model="D:\Yolo Dataset\YOLO_TrainingScriptsuns	eacher_yolo11x_seg\weightsest.pt"   data="D:\Yolo Datasetottle_cap_sdp.v7i.yolov11\data.yaml"   split=test   imgsz=640   batch=16   device=0   project="D:\Yolo Dataset\YOLO_TrainingScriptsuns"   name=val_teacher_test
```

#### 2. Batch Inference on Test Directory
```bash
# Run prediction on full test set and save annotated visuals
yolo segment predict   model="D:\Yolo Dataset\YOLO_TrainingScriptsuns\student_yolo11m_distilled\weightsest.pt"   source="D:\Yolo Datasetottle_cap_sdp.v7i.yolov11	est\images"   conf=0.25   iou=0.7   imgsz=640   device=0   save=True   project="D:\Yolo Dataset\YOLO_TrainingScriptsuns"   name=predict_student_test
```

#### 3. Model Export via CLI
```bash
# Export to standard ONNX with simplification
yolo export   model="D:\Yolo Dataset\YOLO_TrainingScriptsuns\student_yolo11m_distilled\weightsest.pt"   format=onnx   imgsz=640   dynamic=True   simplify=True

# Export to Half-Precision (FP16) ONNX
yolo export   model="D:\Yolo Dataset\YOLO_TrainingScriptsuns\student_yolo11m_distilled\weightsest.pt"   format=onnx   half=True   imgsz=640
```

#### 4. Speed & Latency Benchmarking
```bash
# Benchmark model latency and throughput across formats
yolo benchmark   model="D:\Yolo Dataset\YOLO_TrainingScriptsuns\student_yolo11m_distilled\weightsest.pt"   data="D:\Yolo Datasetottle_cap_sdp.v7i.yolov11\data.yaml"   imgsz=640   device=0
```

---

### Python API Programmatic Evaluation

Use this script to programmatically extract and print metrics:

```python
from ultralytics import YOLO

weights_path = r"D:\Yolo Dataset\YOLO_TrainingScriptsuns\student_yolo11m_distilled\weightsest.pt"
data_yaml    = r"D:\Yolo Datasetottle_cap_sdp.v7i.yolov11\data.yaml"

model = YOLO(weights_path)

# Run validation on the test split
metrics = model.val(
    data=data_yaml,
    split='test',
    imgsz=640,
    batch=16,
    device=0,
    plots=True
)

# Extract and display key performance indicators
print("
================ EVALUATION METRICS ================")
print(f"Bounding Box mAP@50     : {metrics.box.map50 * 100:.2f}%")
print(f"Bounding Box mAP@50-95  : {metrics.box.map * 100:.2f}%")
print(f"Segmentation Mask mAP@50: {metrics.seg.map50 * 100:.2f}%")
print(f"Segmentation Mask mAP@50-95: {metrics.seg.map * 100:.2f}%")
print(f"Precision (Box)         : {metrics.box.mp * 100:.2f}%")
print(f"Recall (Box)            : {metrics.box.mr * 100:.2f}%")
print(f"Inference Latency       : {metrics.speed['inference']:.2f} ms/image")
print(f"Pre/Post Processing     : {metrics.speed['preprocess']:.2f} / {metrics.speed['postprocess']:.2f} ms")
print("====================================================")
```

---

## 5. Troubleshooting & Best Practices

### 1. Handling CUDA Out of Memory (OOM) Errors
During student distillation (`05_train_student_distill.py`):
- Reduce batch size: Change `batch=16` to `batch=8` or `batch=4`.
- Adjust image resolution: Set `imgsz=512`.
- Ensure `amp=True` is enabled to take advantage of FP16 mixed precision.

### 2. Checkpoint Auto-Resume After Interruption
If a training job is killed or terminated:
- Simply execute the training script again:
  ```bash
  python 05_train_student_distill.py
  ```
- The built-in checkpoint detector automatically detects `last.pt` and resumes from the exact epoch where it was interrupted.

### 3. Verification of Teacher Weight Freezing
During knowledge distillation, teacher weights must strictly remain frozen:
- Script `05_train_student_distill.py` enforces this with:
  ```python
  teacher.eval()
  for param in teacher.parameters():
      param.requires_grad = False
  ```
- Only student parameters and the projection adapters receive gradient updates.

---
*Guide compiled for Capstone Project Team 25 - Automated Quality Inspection System.*
