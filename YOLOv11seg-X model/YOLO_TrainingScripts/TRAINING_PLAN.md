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
| **Goal** | Train a high-accuracy Large Teacher model, then distill its knowledge into a compact Nano Student model for real-time edge deployment |

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

## Phase 4: Nano Student Training (yolo11n-seg) via Knowledge Distillation

### 4.1 Model Architecture: yolo11n-seg

| Parameter | Value |
|---|---|
| **Backbone** | CSPDarknet (nano: width=0.25, depth=0.33 multipliers) |
| **Neck** | PANet (lightweight version) |
| **Segmentation Head** | 32 prototype masks (same structure as teacher) |
| **Parameters** | ~2.9M (vs Teacher's 56.9M = ~20x smaller) |
| **Inference Speed** | ~1.8ms/image on GPU (vs Teacher's ~18ms = ~10x faster) |
| **Model Size** | ~12MB (vs Teacher's ~114MB) |

### 4.2 Knowledge Distillation: Two-Level Strategy

The Nano Student is trained with a combined loss using:
1. **Ground Truth Loss** — supervised by annotated polygon labels
2. **Response-Level Distillation** — learning from Teacher's output logits (KL Divergence)
3. **Feature-Level Distillation** — learning from Teacher's intermediate PANet features (MSE)

#### Distillation Loss A: KL Divergence on Output Logits

```
L_KD = T^2 * sum( p_teacher(c|x;T) * log( p_teacher(c|x;T) / p_student(c|x;T) ) )
```

Where temperature-softened probabilities are computed as:

```
p_softened(c) = exp(logit_c / T) / sum_k( exp(logit_k / T) )
```

Parameters:
- `T = 4` (temperature — higher = softer distributions)
- `T^2` normalization factor preserves gradient magnitude scale
- At T=4, damaged_cap probability: 0.72 -> 0.48 (softer), good_cap: 0.15 -> 0.24 (higher)

This means the student sees the teacher's uncertainty, not just the winning class.

#### Distillation Loss B: Feature MSE on PANet Neck Outputs

```
L_feat = sum_l( || F_teacher_l - Adapter_l(F_student_l) ||^2_F )
```

Where:
- `F_teacher_l` = Teacher's feature map at PANet neck layer l
- `Adapter_l` = Learnable 1x1 convolution to align dimensions (teacher channels -> student channels)
- `||.||^2_F` = Frobenius norm (sum of squared element differences)

This forces the Student's feature representations to mimic the Teacher's internal structure at multiple scales.

### 4.3 Total Student Loss Function

```
L_student = alpha * L_gt + beta * L_KD + gamma * L_feat
```

| Component | Formula | Weight | Purpose |
|---|---|---|---|
| Ground Truth Loss | L_gt = L_box + L_cls + L_mask | alpha = 0.4 | Anchors to correct labels |
| KL Divergence (Response KD) | T^2 * KL(p_teacher || p_student) | beta = 0.5 | Transfers class probability knowledge |
| Feature MSE (Feature KD) | ||F_teacher - Adapter(F_student)||^2_F | gamma = 0.1 | Aligns intermediate representations |

### 4.4 Student Training Hyperparameters

```python
student_config = {
    "model":           "yolo11n-seg.pt",  # Pretrained COCO nano weights
    "data":            "data.yaml",
    "epochs":          180,               # Students need more epochs to converge
    "imgsz":           640,
    "batch":           32,                # Larger batch fits: smaller model
    "device":          0,
    "optimizer":       "AdamW",
    "lr0":             0.002,             # Slightly higher LR for smaller model
    "lrf":             0.01,
    "momentum":        0.937,
    "weight_decay":    0.0005,
    "warmup_epochs":   5,
    "patience":        30,
    "cos_lr":          True,
    "label_smoothing": 0.0,               # Off: soft labels already provide smoothing
    "cls":             0.5,
    "box":             7.5,
    "mosaic":          1.0,
    "copy_paste":      0.3,
    "project":         "bottle_cap_seg",
    "name":            "student_yolo11n_distilled",
    "save":            True,
    "plots":           True,
    # Distillation-specific:
    "teacher_weights": "bottle_cap_seg/teacher_yolo11x/weights/best.pt",
    "kd_temperature":  4,
    "kd_alpha":        0.4,               # GT loss weight
    "kd_beta":         0.5,               # KL divergence loss weight
    "kd_gamma":        0.1,               # Feature MSE loss weight
}
```

---

## Phase 5: Evaluation and Comparison

### 5.1 Metrics to Track

| Metric | Teacher (yolo11x-seg) | Student Target | Retention Goal |
|---|---|---|---|
| Box mAP50 | ~0.92 | >= 0.87 | > 95% |
| Mask mAP50 | ~0.90 | >= 0.85 | > 94% |
| Mask mAP50-95 | ~0.72 | >= 0.65 | > 90% |
| Inference ms/img (GPU) | ~18ms | <= 3ms | 6x speedup |
| Model Size (MB) | ~114MB | <= 12MB | 10x smaller |
| Parameters | 56.9M | 2.9M | 20x fewer |

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
teacher = YOLO("bottle_cap_seg/teacher_yolo11x/weights/best.pt")
teacher.val(data="data.yaml", split="test")

# Student evaluation
student = YOLO("bottle_cap_seg/student_yolo11n_distilled/weights/best.pt")
student.val(data="data.yaml", split="test")
```

---

## Phase 6: Export for Production Deployment

```python
student = YOLO("bottle_cap_seg/student_yolo11n_distilled/weights/best.pt")

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
| 05 | `05_train_student_distill.py` | Train yolo11n-seg student with GT + KL + Feature distillation loss |
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
