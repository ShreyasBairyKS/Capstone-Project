# Architecture — Himalaya NSC Label Defect Detection

## Problem Statement

We inspect printed labels on **Himalaya Winter Defense Moisturizing Cream (50 ml)** tubes.
The labels are scanned as flat unwrapped images — each image is roughly **1504 × 8000 pixels** in BMP format.
The goal is to automatically flag defects like torn print, smudges, missing text, ink blobs, or wrinkles before the product leaves the line.

We had **80 good images** and **42 defective images** to work with. No pre-built dataset, no prior annotations.

---

## Why a Two-Stage Pipeline

The label is long and has four visually distinct regions. Rather than trying to detect defects across the entire 8000px image at once (which is computationally wasteful and inaccurate), we break the problem into two steps:

1. **Find where each region is** — using an object detector
2. **Check if that region looks normal** — using an anomaly model

This makes each model simpler and more focused. The detector only needs to locate boxes, and the anomaly model only sees one region at a time.

---

## Stage 1 — Region Detection (YOLO11m + SAHI)

### The 4 Regions (ROIs)

| ROI | Content |
|-----|---------|
| ROI_1 | Himalaya logo + product name |
| ROI_2 | Ingredient text (Jojoba Oil, Wheat Germ, Almond Oil) |
| ROI_3 | Address, regulatory info, MFG/EXP dates, barcode |
| ROI_4 | "3 Way Care" graphic with icons |

Since the label wraps around a cylindrical tube, **each ROI appears twice per image**. The detector always picks the complete (uncut) occurrence.

### Why YOLO11m

We chose **YOLO11m** (medium variant) over older models like YOLOv8 for slightly better feature extraction at a similar parameter count. The medium variant was selected deliberately — small is too coarse for our task, and large adds training time without a meaningful accuracy gain on a 4-class problem.

### The Resolution Problem and SAHI

A standard YOLO run on an 8000px image with `imgsz=1024` downscales the image by ~8×. At that scale, the ROI label regions — which are relatively thin horizontal strips — effectively disappear. The detector was missing all 4 ROIs on most images when trained this way.

The fix was **SAHI (Slicing Aided Hyper Inference)**:
- The 8000px image is sliced into overlapping **1504×1504 tiles** (20% overlap)
- YOLO runs on each tile at full resolution
- Predictions from all tiles are merged using NMS

For training, a companion script (`slice_dataset.py`) applies the same slicing to the training images and recalculates all bounding box coordinates accordingly, so the model is trained and tested under identical conditions.

---

## Stage 2 — Anomaly Classification (EfficientAD-Medium)

### Why Anomaly Detection Instead of Classification

We only had images labeled as "good" or "bad" — we did not have defect-type labels (tear, smudge, etc.) for most images. Anomaly detection is the right fit here because:
- The model trains only on **good images**, learning what "normal" looks like
- At test time it scores how much the input deviates from normal
- Any deviation above a threshold → defect flagged

This also means if a new defect type appears that we've never seen before, the model will still catch it as long as it looks different from the normal label.

### EfficientAD — How It Works

EfficientAD uses a **student-teacher** approach combined with an **autoencoder**:

- A pre-trained **teacher** network (EfficientNet-based) extracts features from a good image crop
- A **student** network learns to mimic the teacher's features on good images during training
- An **autoencoder** learns to reconstruct good images during training
- At inference, on a defective region:
  - The student fails to mimic the teacher → high prediction error
  - The autoencoder fails to reconstruct the defect cleanly → high reconstruction error
- Both errors are combined pixel-by-pixel into an **anomaly map** (heatmap)
- The mean of the anomaly map is the final anomaly score

### Why EfficientAD-Medium over alternatives

We evaluated the options:
- **EfficientAD-Small**: faster but ~6% lower detection accuracy on our test set
- **EfficientAD-Medium**: our choice — good balance of speed and accuracy
- **PatchCore**: higher accuracy but uses a memory bank that grows with dataset size, making it too slow for real-time use and unsuitable for edge deployment

### Input Size

EfficientAD's native input is **256×256 pixels**. The cropped ROI region from the 8000px image is resized to this before being passed to the model.

### Threshold Calibration

After training, the model is run on all known bad crops. The threshold is set at the **5th percentile of bad-image scores** — this ensures at least 95% of real defects are caught. This is a deliberate tradeoff: we accept a small number of false alarms to avoid missing real defects.

Thresholds are stored per-ROI in `himalaya_label_detection/config/roi_thresholds.json`.

---

## Grayscale Experiment

During testing, ROI_1, ROI_2, and ROI_3 models trained on color crops were flagging large white/bright regions as anomalies even on good images. The white background was creating a spurious feature signal.

We retrained the same EfficientAD-Medium architecture on **grayscale versions** of the crops (converted to grayscale then stacked to 3 channels so EfficientAD's input dimensions stay unchanged). The results:

| ROI | Color Recall | Grayscale Recall | FP (color) | FP (grayscale) |
|-----|-------------|-----------------|-----------|----------------|
| ROI_1 | ~79% | **93.9%** | 11 | 6 |
| ROI_2 | ~86% | **92.9%** | 7 | 6 |
| ROI_3 | ~74% | **93.5%** | 8 | 13 |

Grayscale training removed the color bias and improved recall significantly. The grayscale models are now the primary models for ROI_1, ROI_2, ROI_3.

ROI_4 has a different issue (annotation quality) and is being retrained separately.

---

## Dataset Structure

```
data/rois/             — original color crops (good/ and bad/ per ROI)
data/gray_scale_rois/  — grayscale versions of the same crops
data/annotations/      — YOLO-format bounding box annotations for defect locations
data/yolo_dataset/     — full images + YOLO labels for detector training
data/yolo_dataset_sliced/ — 1504×1504 tiles generated by slice_dataset.py

models/rois/           — EfficientAD checkpoints trained on color crops
models/gray_scale_rois/— EfficientAD checkpoints trained on grayscale crops
```

---

## Key Numbers

| Component | Detail |
|-----------|--------|
| Input image size | ~1504 × 8000 px, BMP |
| YOLO tile size | 1504 × 1504 px (20% overlap) |
| YOLO model | YOLO11m |
| Anomaly model | EfficientAD-Medium |
| Anomaly input size | 256 × 256 px |
| Training images (good) | 80 per ROI |
| Test images | 120 total (78 good, 42 bad) |
| Recall on crops (grayscale) | >92% for ROI_1, 2, 3 |
| Threshold strategy | 5th percentile of bad scores |
| GPU used | NVIDIA RTX A5000 (24 GB) |
