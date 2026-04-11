# VisionFood QAI — ML & Backend Deep Dive

---

## Purpose

This document explains the deep learning pipeline end-to-end: why each model was chosen, what it does internally, how they connect, and how to interpret their outputs. Intended as a reference for team members, reviewers, and for explaining the system in the capstone viva.

---

## 1. The Four Defect Classes

| Class | Visual Signature | Why It's Hard |
|-------|-----------------|---------------|
| `improper_filling` | Air gap at top, sunken seal centre, visible underfill line | Fill variance is subtle; varies by product SKU |
| `packaging_damage` | Dent, crack, tear, broken seal, crease | Highly varied appearance; overlap with normal wear |
| `label_misalignment` | Skewed label edge, label wrinkle, partial label, off-centre | Misalignment is a relative measurement requiring reference |
| `surface_contamination` | Dark spot, stain patch, foreign particle on surface | Some contamination is translucent; lighting-dependent |

---

## 2. YOLOv11 — Object Detection

### What It Does
YOLOv11 takes a 640×640 RGB image and outputs the location and class of every defect present. It answers: **"Is there a defect, where is it, and which class is it?"**

### Architecture Internals
```
Input: 640×640×3 float32 tensor (normalised [0,1])
   │
   ▼
Backbone: C3k2 blocks + SPPF
   ├── C3k2 = Cross-Stage Partial with 2 kernels (efficient multi-scale feature extraction)
   ├── SPPF = Spatial Pyramid Pooling Fast (captures multi-scale context in one pass)
   └── Progressive downsampling: 640→320→160→80→40→20 feature map sizes
   │
   ▼
Neck: C2PSA + Feature Pyramid Network (FPN)
   ├── C2PSA = Cross-Stage Partial with Spatial Attention (attends to defect regions)
   ├── FPN upsamples and concatenates deep + shallow features
   └── Outputs 3 prediction heads at scales: 80×80, 40×40, 20×20
   │
   ▼
Detection Head (3 heads × 8400 anchors total)
   ├── Each anchor predicts: [x_center, y_center, width, height, confidence, class_id]
   ├── Decoupled head: separate branches for box regression and classification
   └── Total output: (1, 8400, 6) raw predictions
   │
   ▼
Post-processing
   ├── Confidence threshold filter: drop all anchors with conf < 0.40
   ├── Non-Maximum Suppression (NMS): remove overlapping boxes (IoU threshold 0.45)
   └── Final output: List[Detection] with (class, confidence, normalised bbox)
```

### Why YOLOv11 (not YOLOv8 or ResNet)?
- YOLOv11n has **22% fewer parameters** than YOLOv8n with equivalent or higher mAP
- C3k2 block is more parameter-efficient than C2f for small defect localisation
- Native ONNX export path — no model surgery required
- Real-time capable: 1.5ms on Jetson (INT8), ~40–80ms CPU ONNX

### Training Configuration
```
Pretrained weights: yolov11n.pt (COCO-pretrained)
Dataset: 1,500+ annotated food defect images
Epochs: 100 (early stopping patience=20)
Batch size: 16
Image size: 640×640
Optimizer: AdamW (lr=0.001, weight_decay=0.0005)
LR schedule: Cosine annealing
Loss = λ_box × L_CIoU + λ_cls × L_BCE + λ_dfl × L_DFL
  where λ_box=7.5, λ_cls=0.5, λ_dfl=1.5 (Ultralytics defaults)
```

### Augmentations Applied During Training
| Augmentation | Value | Purpose |
|-------------|-------|---------|
| Mosaic | p=0.5 | Combines 4 images; increases small-object recall |
| MixUp | α=0.2, p=0.3 | Blends two images; improves generalisation |
| Horizontal flip | p=0.5 | Rotation-invariance for symmetric defects |
| Vertical flip | p=0.3 | Covers overhead-camera orientation variants |
| Brightness/contrast | ±30%, p=0.6 | Factory lighting variation simulation |
| Gaussian noise | var 10–50, p=0.4 | Camera sensor noise robustness |
| Rotation | ±15°, p=0.4 | Label skew, slight conveyor angle variation |

---

## 3. EfficientViT-M5 — Fine-Grained Classifier

### What It Does
EfficientViT receives the cropped bounding box region from YOLOv11 and outputs a **refined 4-class probability distribution**. It answers: **"Given this defect crop, which class is it most precisely?"**

### Why a Second Model?
YOLOv11's classification head is trained jointly with detection. Its classification of the crop is less accurate than a dedicated classifier trained specifically on cropped defect patches. EfficientViT improves top-1 accuracy from ~88% (YOLO-only) to ~97% (YOLO + ViT).

### Architecture Internals
```
Input: 224×224×3 cropped defect patch
   │
   ▼
Stem: 3×3 Conv → BN → ReLU (initial feature extraction)
   │
   ▼
EfficientViT Stages (M5 = 5 stages)
   ├── Each stage: Local Conv branch + Linear Attention branch
   ├── Linear Attention: Q×K attention computed as (Q×K^T)×V
   │     O(N) not O(N²) — key efficiency gain over standard ViT
   ├── Multi-Scale Learning: stride-2 downsampling between stages
   └── Global average pooling → 512-dim feature vector
   │
   ▼
Classification Head
   ├── Linear(512, 128) → ReLU → Dropout(0.3)
   ├── Linear(128, 4)
   └── Softmax → [p_filling, p_damage, p_label, p_contamination]
```

### Training Configuration
```
Pretrained: efficientvit_m5 from timm (ImageNet weights)
Loss: Focal Loss (gamma=2.0, alpha=0.25) — handles class imbalance
Optimizer: AdamW (lr=3e-4)
Schedule: Cosine annealing (T_max=50 epochs)
Label smoothing: 0.1
Batch size: 32
Input: 224×224 cropped patches from YOLOv11 detections + 10% padding
```

### Why Focal Loss?
Standard cross-entropy weighs all samples equally. Focal loss down-weights easy examples (clean products correctly classified with high confidence) and up-weights hard examples (subtle contamination, marginal filling). For a 4-class dataset inevitably imbalanced toward label misalignment (most common defect), focal loss is critical for contamination class recall.

---

## 4. MC Dropout — Uncertainty Quantification

### The Problem It Solves
A neural network returns a softmax probability — but this probability is **not a reliable confidence estimate**. A model can output 0.95 confidence on an image it has never seen before (out-of-distribution). MC Dropout provides a calibrated uncertainty estimate.

### How It Works
```
Normal inference:
  model.eval() → dropout layers OFF → single deterministic output

MC Dropout inference:
  enable_dropout(model) → dropout layers ACTIVE during inference
  Run 20 forward passes on the same input
  
  Each pass: dropout randomly zeros different neurons → slightly different prediction
  
  predictions = [softmax(model(x)) for _ in range(20)]
  
  mean  = np.mean(predictions, axis=0)   # Best prediction estimate
  std   = np.std(predictions,  axis=0)   # Uncertainty (spread of predictions)
  
  confidence_interval = [mean - 2×std,  mean + 2×std]
```

### Interpreting the Output

| `uq_std` Value | Meaning | Action |
|---------------|---------|--------|
| < 0.05 | Very confident — well-represented in training data | Accept verdict as-is |
| 0.05–0.12 | Moderate confidence — some uncertainty | Accept, flag for monitoring |
| > 0.12 | High uncertainty — possibly out-of-distribution | Escalate or human review |

### The Confidence Threshold Decision Logic

```
mean_confidence = uq_result.mean_confidence
std_confidence  = uq_result.std_confidence

if mean_confidence ≥ 0.85 and std_confidence < 0.12:
    # High confidence, low uncertainty: direct decision
    verdict = "FAIL" if any detections else "PASS"

elif 0.60 ≤ mean_confidence < 0.85:
    # Medium confidence: FAIL but flag for log monitoring
    verdict = "FAIL", escalated = True

elif 0.45 ≤ mean_confidence < 0.60:
    # Low confidence: human review required before action
    verdict = "ESCALATE", escalated = True

else:
    # Very low confidence: manual only
    verdict = "REVIEW", escalated = True
```

---

## 5. REMEDY Severity Engine

### Purpose
Not every defect is equally serious. A cosmetic label skew and a seal failure are both detected as defects — but rejecting them both with the same finality is wasteful and commercially wrong. The REMEDY engine grades and routes.

### Severity Score Formula

```
raw_score = (0.35 × area_score)
          + (0.15 × confidence_uncertainty_score)
          + (0.40 × class_risk_score)
          + (0.10 × attempt_penalty)

where:
  area_score               = min(1.0, defect_area_fraction × 20)
  confidence_uncertainty   = 1.0 - detection_confidence
  class_risk_score         = CLASS_SEVERITY_WEIGHTS[class_name]
  attempt_penalty          = min(1.0, attempt_count × 0.30)

CLASS_SEVERITY_WEIGHTS:
  surface_contamination: 0.90   (food safety risk)
  packaging_damage:      0.60   (integrity risk)
  improper_filling:      0.45   (commercial risk)
  label_misalignment:    0.30   (regulatory/cosmetic)
```

### Grade Thresholds and Actions

```
raw_score < 0.30  →  S1 (Minor)    → Remediable
0.30–0.55         →  S2 (Moderate) → Remediable
0.55–0.75         →  S3 (Serious)  → Reject (case-by-case)
≥ 0.75            →  S4 (Critical) → Hard reject

Remediation actions (capstone: logged only, no hardware):
  label_misalignment  + S1/S2  → action: RELABEL   (Station A)
  improper_filling    + S1/S2  → action: REFILL     (Station B)
  packaging_damage    + S1/S2  → action: REPACK     (Station C)
  surface_contamination + S1   → action: CLEAN      (Station C)
  all S3/S4                    → action: REJECT
```

---

## 6. Full Pipeline Data Flow

```
Frame: np.ndarray (BGR, any resolution)
    │
    ▼ inference/preprocessor.py::letterbox(frame, 640)
    │
    [640×640 normalised NCHW tensor]
    │
    ▼ YOLOv11TRTDetector.detect()  [ONNX Runtime session]
    │
    [List[Detection]] ← post-processed, NMS applied
    │
    ├── Empty? → Verdict = PASS, skip to publish
    │
    ▼ Top detection bbox → crop_with_padding(frame, detection, pad=0.10)
    │
    [Cropped patch, resized to 224×224]
    │
    ▼ EfficientViTClassifier.classify()  [ONNX Runtime session]
    │
    [cls_result: (class_id, confidence, all_probs)]
    │
    ▼ MCDropoutUQ.predict_with_uncertainty()  [20 passes, dropout ON]
    │
    [UQResult: mean, std, ci_low, ci_high, is_uncertain, escalate_required]
    │
    ▼ EdgeInferencePipeline._apply_verdict_logic()
    │
    [verdict: PASS | FAIL | ESCALATE | REVIEW]
    │
    ├── PASS? → skip REMEDY
    │
    ▼ SeverityScorer.score(detection, attempt_count=0)
    │
    [SeverityResult: raw_score, grade, recommended_action]
    │
    ▼ TriageRouter.route(detection, severity_result)
    │
    [TriageDecision: action, station, is_remediable, reason]
    │
    ▼ InspectionResult assembled (all fields populated)
    │
    ├── SQLAlchemy: Inspection + Defect + RemediationAction written to DB
    │
    └── Redis XADD "inspections:live" → WebSocket fan-out → Dashboard
```

---

## 7. Model Export Pipeline (ONNX)

### Why ONNX?
ONNX (Open Neural Network Exchange) decouples model training (PyTorch) from model serving (ONNX Runtime). Benefits:
- Hardware agnostic: same `.onnx` file runs on CPU, CUDA GPU, or Intel NPU
- No PyTorch required at inference time — smaller Docker image
- Enables future TensorRT INT8 optimisation without retraining

### Export Steps

**YOLOv11 → ONNX**
```bash
# Ultralytics handles this natively
yolo export model=models/yolov11n_best.pt format=onnx half=True imgsz=640
# Output: models/yolov11n_best.onnx
```

**EfficientViT → ONNX**
```python
import torch
import timm

model = timm.create_model("efficientvit_m5", pretrained=False, num_classes=4)
model.load_state_dict(torch.load("models/efficientvit_m5_best.pt"))
model.eval()

dummy = torch.randn(1, 3, 224, 224)
torch.onnx.export(
    model, dummy,
    "models/efficientvit_m5_best.onnx",
    input_names=["input"],
    output_names=["logits"],
    dynamic_axes={"input": {0: "batch_size"}},
    opset_version=17
)
```

### ONNX Runtime Session Setup
```python
import onnxruntime as ort

session = ort.InferenceSession(
    "models/yolov11n_best.onnx",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
)
# ONNX Runtime auto-selects CUDA if available, falls back to CPU transparently
```

---

## 8. MLflow Experiment Tracking

Every training run is logged with:
```python
import mlflow

with mlflow.start_run(run_name="yolov11n_v1.0.0"):
    mlflow.log_params({
        "epochs": 100, "batch": 16, "imgsz": 640, "optimizer": "AdamW"
    })
    mlflow.log_metrics({
        "map50": 0.832, "map50_95": 0.741,
        "precision": 0.851, "recall": 0.803
    })
    mlflow.log_artifact("models/yolov11n_best.onnx")
    mlflow.log_artifact("results/confusion_matrix.png")
```

View experiments: `mlflow ui --port 5000` → `http://localhost:5000`

---

## 9. Key Performance Metrics Reference

| Metric | Definition | Target |
|--------|-----------|--------|
| mAP@50 | Mean Average Precision at IoU=0.50 threshold | ≥ 0.80 |
| mAP@50-95 | mAP averaged over IoU 0.50–0.95 | ≥ 0.65 |
| Precision | TP / (TP + FP) — what fraction of detections are real defects | ≥ 0.85 |
| Recall | TP / (TP + FN) — what fraction of real defects are detected | ≥ 0.80 |
| F1 | 2 × Precision × Recall / (Precision + Recall) | ≥ 0.82 |
| FNR | False Negative Rate = FN / (TP + FN) | < 3% |
| Top-1 Accuracy (ViT) | Correct class / total classified crops | ≥ 95% |

**Critical rule:** FNR (False Negative Rate) is the most important metric. A missed defect that reaches a consumer is far more damaging than a false positive that stops a good product.
