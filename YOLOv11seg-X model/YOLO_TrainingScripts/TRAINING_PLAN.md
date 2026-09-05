# YOLO Segmentation Training Plan: Teacher to Student Knowledge Distillation
## Bottle Cap Defect Detection | 6-Class Instance Segmentation

---

## Project Overview

| Property | Details |
|---|---|
| **Dataset** | `bottle_cap_sdp.v7i.yolov11` |
| **Classes** | `damaged_cap`, `good_cap`, `misplaced_cap`, `no_cap`, `open_cap`, `wet_cap` |
| **Total Images** | 8,530 (Train: 7,464 · Valid: 712 · Test: 354) |
| **Label Format** | YOLO Polygon Segmentation (multi-vertex contours) |
| **Task** | Instance Segmentation (mask + bounding box prediction) |
| **Goal** | Train a high-accuracy Large Teacher model (yolo11x-seg), then distill its knowledge into a Medium Student model (yolo11m-seg) via Feature-Based KD for real-time edge deployment |

---

## Phase 1: Dataset Preparation

### 1.1 Ensure YOLO-Ready Flat Structure

Merge class subfolders back into unified `images/` and `labels/` folders per split.

```
bottle_cap_sdp.v7i.yolov11/
├── train/
│   ├── images/     <- All 7,464 training images
│   └── labels/     <- All 7,464 YOLO polygon labels
├── valid/
│   ├── images/     <- All 712 validation images
│   └── labels/     <- All 712 YOLO polygon labels
└── test/
    ├── images/     <- All 354 test images
    └── labels/     <- All 354 YOLO polygon labels
```

Run: `python merge_to_yolo_format.py` (located at `d:\Yolo Dataset\merge_to_yolo_format.py`)

### 1.2 Verify data.yaml

```yaml
path: D:/Yolo Dataset/bottle_cap_sdp.v7i.yolov11
train: train/images
val:   valid/images
test:  test/images

nc: 6
names: ['damaged_cap', 'good_cap', 'misplaced_cap', 'no_cap', 'open_cap', 'wet_cap']
```

### 1.3 Dataset Health Checks

- [x] Every image has a matching .txt polygon label (verified via `verify_dataset_labels.py`)
- [ ] Check class distribution for imbalance (good_cap: 4,242 vs misplaced_cap: 522 — significant imbalance!)
- [ ] Apply class-weighted loss to compensate for imbalance (see Phase 2 hyperparameters)

**Class Distribution Summary:**

| Class | Train | Valid | Test |
|---|---|---|---|
| good_cap | 4,242 | 405 | 201 |
| wet_cap | 828 | 79 | 39 |
| open_cap | 768 | 65 | 38 |
| damaged_cap | 648 | 62 | 31 |
| no_cap | 645 | 61 | 30 |
| misplaced_cap | 327 | 40 | 15 |

---

## Phase 2: Teacher Model Training (yolo11x-seg)

### 2.1 Rationale for the Teacher

The Teacher model (`yolo11x-seg`, Extra-Large variant) is a high-capacity model used to:
1. Learn precise segmentation mask contours of all cap conditions.
2. Generate soft prediction distributions (logits) that encode richer per-pixel confidence than hard binary labels.
3. Provide **soft labels** (dark knowledge) to train the Nano Student more effectively than ground-truth labels alone.

### 2.2 Model Architecture: yolo11x-seg

| Parameter | Value |
|---|---|
| **Backbone** | CSPDarknet (C3 + ELAN blocks, extra-large width/depth multipliers) |
| **Neck** | PANet (Path Aggregation Network — multi-scale feature fusion) |
| **Segmentation Head** | Prototype-based mask decoder (32 mask prototypes per image) |
| **Parameters** | ~56.9M |
| **Input Resolution** | 640x640 |
| **Output per object** | [x, y, w, h, class_conf x 6, mask_coeffs x 32] |

### 2.3 Teacher Training Hyperparameters

```python
teacher_config = {
    "model":          "yolo11x-seg.pt",  # Pretrained COCO weights
    "data":           "data.yaml",
    "epochs":         150,               # Early stopping will halt if no improvement
    "imgsz":          640,
    "batch":          8,                 # Reduce to 4 if GPU OOM on <8GB VRAM
    "device":         0,                 # GPU 0 (CUDA)
    "optimizer":      "AdamW",
    "lr0":            0.001,             # Initial learning rate
    "lrf":            0.01,              # Final LR = lr0 x lrf (cosine decay)
    "momentum":       0.937,
    "weight_decay":   0.0005,
    "warmup_epochs":  5,
    "patience":       25,                # Early stopping patience
    "cos_lr":         True,              # Cosine LR annealing
    "label_smoothing":0.05,              # Regularization for classification head
    "cls":            0.5,               # Class loss weight
    "box":            7.5,               # Box regression loss weight
    "dfl":            1.5,               # Distribution Focal Loss weight
    "hsv_h":          0.015,
    "hsv_s":          0.7,
    "hsv_v":          0.4,
    "degrees":        10.0,              # Rotation augmentation
    "translate":      0.1,
    "scale":          0.5,
    "flipud":         0.3,               # Important for tilted/misplaced caps
    "fliplr":         0.5,
    "mosaic":         1.0,               # 4-image mosaic patches
    "mixup":          0.15,
    "copy_paste":     0.3,               # Copy-paste (excellent for segmentation)
    "project":        "bottle_cap_seg",
    "name":           "teacher_yolo11x",
    "save":           True,
    "plots":          True
}
```

### 2.4 Loss Functions: Teacher Training

The YOLO11 segmentation model uses three combined loss components:

#### A. Bounding Box Regression Loss: DFL + CIoU

**Distribution Focal Loss (DFL)** — precise bounding box edge localization:

```
L_DFL = sum( -((y_t+1 - y_j) * log(P_j) + (y_j - y_t) * log(P_{j+1})) )
```

- `y_t` = target box coordinate value
- `P_j` = predicted probability at discrete coordinate bin j

**Complete IoU (CIoU)** — box geometry overlap and shape consistency:

```
L_CIoU = 1 - IoU + rho^2(b, b_gt)/c^2 + alpha*nu
```

- `IoU` = Intersection over Union of predicted vs ground-truth box
- `rho(b, b_gt)` = Euclidean distance between box centers
- `c` = diagonal length of enclosing box
- `alpha*nu` = aspect ratio consistency penalty

**Combined Box Loss Weight**: `box = 7.5`

#### B. Classification Loss: Binary Cross-Entropy with Label Smoothing

```
L_cls = -sum( y_s * log(p) + (1 - y_s) * log(1 - p) )
```

Where `y_s = (1 - epsilon) * y + epsilon/K`
(label-smoothed target, epsilon=0.05, K=6 classes)

**Classification Loss Weight**: `cls = 0.5`

#### C. Segmentation Mask Loss: BCE on Prototype Masks

YOLO11-seg uses mask prototype decomposition:

```
mask_pred = sum_k( coeff_k * prototype_k )    for k = 1..32
```

Mask BCE loss:

```
L_mask = -sum( m_gt * log(sigmoid(mask_pred)) + (1 - m_gt) * log(1 - sigmoid(mask_pred)) )
```

#### D. Total Teacher Loss

```
L_teacher = box * L_box + cls * L_cls + L_mask
           = 7.5 * L_DFL_CIoU + 0.5 * L_cls + L_mask
```

---

## Phase 3: Teacher Evaluation and Soft Label Generation

### 3.1 Benchmark the Teacher Model

```python
metrics = teacher_model.val(split="test")
# Target metrics for a production-ready teacher:
# Box  mAP50     > 0.90
# Mask mAP50     > 0.88
# Mask mAP50-95  > 0.70
```

### 3.2 Generate Soft Labels (Pseudo-Labels / Dark Knowledge)

Use the trained Teacher to run inference on the full training set and save its predicted class logits and mask coefficients as soft labels.

These soft logits encode more nuanced probability distributions than binary 0/1 hard labels.

Example comparison:
- Hard label for good_cap:    [0, 1, 0, 0, 0, 0]
- Soft label from teacher:    [0.02, 0.94, 0.01, 0.01, 0.01, 0.01]  (confident)
- Soft label for damaged_cap: [0.72, 0.15, 0.04, 0.05, 0.03, 0.01]  (captures uncertainty)

The soft label for damaged_cap reveals the model's understanding that there is some similarity to good_cap (0.15) — this inter-class relationship is the "dark knowledge" transferred to the student.

---

## Phase 4: Medium Student Training (yolo11m-seg) via Feature-Based Knowledge Distillation

> **Design Decision (finalised):** KD strategy is **Feature-Level Only** per `YOLO11_Feature_KD_Guide.md`.
> Response-level KD (KL Divergence on output logits) has been **removed**.
> Rationale: Minute defects occupy <1% of pixel real estate; feature distillation directly
> aligns the student's spatial attention at C3k2, SPPF, and C2PSA layers — more effective
> than matching final classification outputs for micro-defect detection.

### 4.1 Model Architecture: yolo11m-seg

| Parameter | Value |
|---|---|
| **Backbone** | CSPDarknet (medium: width=0.50, depth=0.67 multipliers) |
| **Neck** | PANet (medium version) |
| **Segmentation Head** | 32 prototype masks (same structure as teacher) |
| **Parameters** | ~27.3M (vs Teacher's 56.9M = ~2x smaller) |
| **Inference Speed** | ~6ms/image on GPU (vs Teacher's ~18ms = ~3x faster) |
| **Model Size** | ~55MB (vs Teacher's ~114MB) |

### 4.2 Knowledge Distillation: Feature-Level Strategy

The Medium Student is trained with a combined loss using:
1. **Ground Truth Loss** — supervised by annotated polygon labels
2. **Feature-Level Distillation** — learning from Teacher's intermediate C3k2, SPPF, and C2PSA feature maps (L2-normalised MSE)

> **Note:** Response-Level Distillation (KL Divergence on output logits) has been removed from this pipeline.

#### Targeted Distillation Layers

| Layer | Location | Purpose |
|---|---|---|
| **C3k2** | Backbone output | Aligns edge/texture features; preserves geometric properties of cap threads and ridges |
| **SPPF** | Backbone→Neck transition | Ensures multi-scale context is preserved alongside micro-defect features |
| **C2PSA** | Neck/Head routing | **Most critical** — directly transfers the teacher's spatial attention gaze to the student |

#### Projector (Hint Layer) Mechanism
Because yolo11m-seg has fewer channels than yolo11x-seg, learnable 1×1 convolutional projectors
map student channels → teacher channels during training.

| Layer | Student Channels | Teacher Channels |
|---|---|---|
| C3k2  | 256 | 512  |
| SPPF  | 512 | 1024 |
| C2PSA | 512 | 1024 |

**Projectors are entirely discarded at inference — zero added latency on edge hardware.**

#### Feature Alignment Loss: L2-Normalised MSE

```
L_layer = || Norm(F_Teacher) - Norm(Projector(F_Student)) ||_2^2
```

Why normalise?
Tiny defects produce sparse, low-magnitude activations. Without normalisation, large uniform
background regions dominate the MSE. Channel-wise L2 normalisation scales all activation
vectors to a unit sphere, forcing the loss to focus on the *pattern* rather than *magnitude*.

### 4.3 Total Student Loss Function

```
L_student = L_gt + alpha(t) * (L_feat_C3k2 + L_feat_SPPF + L_feat_C2PSA)
```

| Component | Formula | Weight | Purpose |
|---|---|---|---|
| Ground Truth Loss | L_gt = L_box + L_cls + L_mask | 1.0 (full weight) | Anchors to correct labels |
| Feature MSE (C3k2) | \|\|Norm(F_T) - Norm(P(F_S))\|\|² | alpha(t) | Aligns backbone texture/edge features |
| Feature MSE (SPPF) | \|\|Norm(F_T) - Norm(P(F_S))\|\|² | alpha(t) | Aligns multi-scale context features |
| Feature MSE (C2PSA) | \|\|Norm(F_T) - Norm(P(F_S))\|\|² | alpha(t) | Transfers spatial attention directly |

**Alpha Warmup Schedule:**

| Epoch Range | Alpha Value | Rationale |
|---|---|---|
| 0 – 20 | 0.05 → 0.50 (linear) | Ramp up gradually to avoid over-mimicry before basic detection is learned |
| 20 – 180 | 0.50 (constant) | Full feature alignment pressure maintained |

### 4.4 Student Training Hyperparameters

```python
student_config = {
    "model":           "yolo11m-seg.pt",  # Pretrained COCO medium weights
    "data":            "data.yaml",
    "epochs":          180,               # Students need more epochs to converge
    "imgsz":           640,
    "batch":           32,                # Fallback: 16 -> 8 -> 4 on OOM
    "device":          0,
    "optimizer":       "AdamW",
    "lr0":             0.001,             # Slightly lower LR for medium vs nano
    "lrf":             0.01,
    "momentum":        0.937,
    "weight_decay":    0.0005,
    "warmup_epochs":   5,
    "patience":        30,
    "cos_lr":          True,
    "label_smoothing": 0.0,               # Off: feature KD provides implicit regularisation
    "cls":             CLS_WEIGHT,        # Inverse-frequency class weighting
    "box":             7.5,
    "fl_gamma":        1.5,               # Focal loss for minority class handling
    "mosaic":          1.0,
    "copy_paste":      0.5,
    "project":         "runs",
    "name":            "student_yolo11m_distilled",
    "save":            True,
    "plots":           True,
    # Distillation-specific:
    "teacher_weights":  "runs/teacher_yolo11x_seg/weights/best.pt",
    "kd_alpha_start":   0.05,             # KD weight warmup start
    "kd_alpha_end":     0.50,             # KD weight warmup end
    "kd_alpha_warmup": 20,               # Warmup duration in epochs
    "kd_layers":       ["C3k2", "SPPF", "C2PSA"],
}
```

---

## Phase 5: Evaluation and Comparison

### 5.1 Metrics to Track

| Metric | Teacher (yolo11x-seg) | Student Target | Retention Goal |
|---|---|---|---|
| Box mAP50 | 0.995 | >= 0.97 | > 97% |
| Mask mAP50 | 0.995 | >= 0.97 | > 97% |
| Mask mAP50-95 | ~0.80 | >= 0.75 | > 93% |
| Inference ms/img (GPU) | ~18ms | <= 8ms | ~2.5x speedup |
| Model Size (MB) | ~114MB | ~55MB | ~2x smaller |
| Parameters | 56.9M | 27.3M | ~2x fewer |

### 5.2 Per-Class Mask mAP50 Comparison

Critical classes to monitor for degradation in the student:

| Class | Teacher mAP50 | Student Target |
|---|---|---|
| good_cap | ~0.95 | >= 0.92 |
| damaged_cap | ~0.91 | >= 0.86 |
| wet_cap | ~0.89 | >= 0.84 |
| open_cap | ~0.90 | >= 0.85 |
| no_cap | ~0.88 | >= 0.83 |
| misplaced_cap | ~0.85 | >= 0.80 |

### 5.3 Evaluation Commands

```python
# Teacher evaluation
teacher = YOLO("runs/teacher_yolo11x_seg/weights/best.pt")
teacher.val(data="data.yaml", split="test")

# Student evaluation
student = YOLO("runs/student_yolo11m_distilled/weights/best.pt")
student.val(data="data.yaml", split="test")
```

---

## Phase 6: Export for Production Deployment

```python
student = YOLO("runs/student_yolo11m_distilled/weights/best.pt")

# ONNX — CPU / OpenVINO / general portable format
student.export(format="onnx", dynamic=True, simplify=True)

# TensorRT FP16 — NVIDIA Jetson / high-speed GPU inspection line
student.export(format="engine", half=True, device=0)

# TensorRT INT8 — maximum throughput (requires calibration dataset)
student.export(format="engine", half=True, int8=True, device=0)

# OpenVINO — Intel NCS2 / Movidius stick
student.export(format="openvino", half=True)
```

---

## Script Roadmap (YOLO_TrainingScripts/)

| # | Script | Purpose |
|---|---|---|
| 01 | `01_merge_dataset.py` | Merge class subfolders to flat YOLO images/labels structure |
| 02 | `02_train_teacher.py` | Train yolo11x-seg teacher model with full augmentation |
| 03 | `03_evaluate_teacher.py` | Validate teacher on test split, print per-class metrics |
| 04 | `04_generate_soft_labels.py` | Run teacher inference to produce soft label files |
| 05 | `05_train_student_distill.py` | Train yolo11m-seg student with GT + Feature KD (C3k2+SPPF+C2PSA, L2-normalised MSE, alpha warmup) |
| 06 | `06_evaluate_and_compare.py` | Side-by-side teacher vs student metrics comparison table |
| 07 | `07_export_student.py` | Export student to ONNX / TensorRT / OpenVINO |

---

## Open Questions Before Execution

> [!IMPORTANT]
> Please confirm the following before proceeding to script creation:
>
> 1. **GPU Specs**: What GPU model and VRAM do you have?
>    - 24GB+ VRAM -> yolo11x-seg (batch=16)
>    - 12-16GB -> yolo11l-seg (batch=8)
>    - 8GB -> yolo11m-seg (batch=4-8)
>
> 2. **Deployment Target**: Where will the nano model run?
>    - NVIDIA Jetson Orin -> TensorRT INT8
>    - Industrial PC with NVIDIA GPU -> TensorRT FP16
>    - CPU Server / Embedded -> ONNX
>
> 3. **Class Imbalance Handling**:
>    - `good_cap` (4,242) vs `misplaced_cap` (522) is 8x imbalance.
>    - Options: (a) apply class weights in loss, (b) oversample minority classes, (c) use focal loss
>
> 4. **Temp Folder Integration**:
>    - The labels moved and reclassified from the Temp folder — should these be merged into the main dataset before training?
