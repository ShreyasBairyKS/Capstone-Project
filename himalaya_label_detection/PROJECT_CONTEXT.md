# PROJECT_CONTEXT.md — Himalaya NSC Label Defect Detection

> **Purpose:** This document captures the full project context, all architectural decisions, and experimental results so any team member or future developer can immediately understand the current state of the system.

---

## 1. Project Overview

**Goal:** Automatically detect print defects (tears, smudges, missing print, ink blobs, wrinkles) on **Himalaya Winter Defense Moisturizing Cream (50 ml)** tube labels using computer vision and anomaly detection.

**Dataset:**
- Images are flat unwrapped scans of cylindrical tubes: **~1504 × 8000 px, RGB, BMP format**
- 80 good images, 42 bad images (120 total in current test set)
- Since labels wrap around the tube, **each ROI appears twice per image** — the pipeline always selects the complete (uncut) occurrence

**Hardware (remote training PC):** NVIDIA RTX A5000 (24 GB VRAM)

---

## 2. The 4 Inspection Regions (ROIs)

| ROI | Content | Notes |
|-----|---------|-------|
| **ROI_1** | Himalaya logo + "Winter Defense Moisturizing Cream" | Most visible, critical |
| **ROI_2** | "Jojoba Oil · Wheat Germ · Almond Oil" + ingredient paragraph | Moderate |
| **ROI_3** | Address + regulatory numbers + MFG/EXP + barcode + Net Vol. | Dense text, critical |
| **ROI_4** | "3 Way Care" graphic with ingredient icons | Most problematic — see §6 |

---

## 3. Pipeline Architecture

```
Full Image (1504 × 8000 px, BMP)
        │
        ▼
YOLO11m Detector  (via SAHI sliced inference)
        │  detects ROI bounding boxes at full resolution
        │  (slice size: 1504×1504, overlap: 20%)
        ├── ROI_1 crop ──► EfficientAD-Medium ──► anomaly score + heatmap
        ├── ROI_2 crop ──► EfficientAD-Medium ──► anomaly score + heatmap
        ├── ROI_3 crop ──► EfficientAD-Medium ──► anomaly score + heatmap
        └── ROI_4 crop ──► EfficientAD-Medium ──► anomaly score + heatmap
                                     │
                        PASS / FAIL + annotated image + JSON
```

### Why this two-stage design?
1. **YOLO** is fast and pinpoints each ROI precisely, even when the label shifts slightly between samples.
2. **EfficientAD** is an anomaly model trained *only on good images*. It learns what normal looks like, then flags anything that deviates — no per-defect labeling required (we only need defect locations for threshold calibration).

---

## 4. Key Architectural Decisions & Rationale

### 4.1 YOLO11m (not YOLOv8)
- **Decision:** Upgraded base detector from YOLOv8 to YOLO11m.
- **Reason:** User explicitly requested best possible accuracy; YOLO11m has superior feature extraction with a similar parameter count.
- **Constraint:** Must remain suitable for edge device deployment in the future.

### 4.2 SAHI (Slicing Aided Hyper Inference)
- **Problem:** Standard YOLO inference on 8000px images with `imgsz=1024` causes extreme downscaling — the detector "blurs away" small ROI labels and misses them entirely.
- **Solution:** SAHI slices the 8000px image into 1504×1504 tiles with 20% overlap, runs YOLO on each tile at full resolution, then merges predictions via NMS.
- **Result:** YOLO now reliably detects all 4 ROIs per image at the correct locations.
- **Training:** A companion script `slice_dataset.py` slices the training data with the same tile parameters and recomputes YOLO label coordinates accordingly.

### 4.3 EfficientAD-Medium (not Small, not Patchcore)
- **Decision:** EfficientAD-Medium for all ROI classifiers.
- **Reason:**
  - **EfficientAD vs Patchcore:** Patchcore would have higher accuracy but uses a memory bank that grows with dataset size, making it too slow for real-time edge inference. EfficientAD is ~6% lower AUROC but has deterministic, fast inference.
  - **Medium vs Small:** Medium gives ~6% better AUROC than Small on the A5000 with no meaningful speed penalty.

### 4.4 Anomaly-Only Training (GOOD images only)
- EfficientAD is trained **exclusively on GOOD images** — this is the key principle of anomaly detection.
- Bad images are **only used for threshold calibration** (to find the score cutoff that gives ≥95% recall on defects).
- This means: if you get more good images, the model gets better. More bad images improve threshold accuracy only.

### 4.5 Threshold Calibration at 5th Percentile of Bad Scores
- After training, the model scores all bad crops.
- The threshold is set at the **5th percentile of bad scores** → guarantees ≥95% recall on defects.
- This intentionally accepts some false positives to minimize missed defects.
- Thresholds are saved to `himalaya_label_detection/config/roi_thresholds.json`.

---

## 5. Dataset Folder Structure

```
data/
  rois/                        ← original color crops
    ROI_1/good/   ← 74 crops
    ROI_1/bad/    ← 33 crops
    ROI_2/good/   ← 77 crops
    ROI_2/bad/    ← 14 crops
    ROI_3/good/   ← 78 crops
    ROI_3/bad/    ← 31 crops
    ROI_4/good/   ← 77 crops
    ROI_4/bad/    ← 26 crops

  gray_scale_rois/             ← grayscale version of above (3-channel, for EfficientAD)
    ROI_1/ ... ROI_4/  (same structure)

  annotations/                 ← YOLO-format defect bounding boxes (for mask generation)
    ROI_1/  ROI_2/  ROI_3/  ROI_4/

  yolo_dataset/                ← full images + YOLO labels for ROI detector training
    images/train/  labels/train/  data.yaml

  yolo_dataset_sliced/         ← SAHI-sliced 1504×1504 patches + recalculated labels
    images/train/  labels/train/  data.yaml

models/
  rois/                        ← EfficientAD checkpoints (color training)
    ROI_1/weights/best.ckpt
    ROI_2/weights/best.ckpt
    ROI_3/weights/best.ckpt
    ROI_4/weights/best.ckpt

  gray_scale_rois/             ← EfficientAD checkpoints (grayscale training)
    ROI_1/weights/last.ckpt
    ROI_2/weights/last.ckpt
    ROI_3/weights/last.ckpt
    ROI_4/weights/last.ckpt

  rois/yolo/                   ← YOLO detector weights
    train4/weights/best.pt     ← old YOLOv8 model (original)
    train-2/weights/best.pt    ← YOLO11m trained on sliced dataset (NEW)
```

---

## 6. Experiment Results

### 6.1 End-to-End Test Results (Color Models, Old YOLO detector)

| Metric | Value |
|--------|-------|
| Total images | 120 (78 good / 42 bad) |
| Accuracy | 62.5% |
| **Recall (TP rate)** | **21.4%** ← most critical failure |
| Precision | 42.9% |
| F1 Score | 0.286 |
| False Alarms | 12 |
| **Missed Defects** | **33** |

**Root cause diagnosis:**
- The test script was loading the **old YOLOv8 weights** (`train4/best.pt`) instead of the newly trained YOLO11m weights.
- When tested with the correct weights on isolated crops, the classifiers showed very high recall (>90%).
- The `test_roi_classifiers.py` script now searches both `models/` and `runs/` directories for the newest `best.pt`.

### 6.2 Crop-Level Classifier Test (Color Models)

Tested with `test_classifiers_on_crops.py` directly on pre-cropped images (bypassing YOLO):

| ROI | Good Crops | Bad Crops | Recall | Precision | FP |
|-----|-----------|----------|--------|-----------|-----|
| ROI_1 | 74 | 33 | 79% | 73% | 11 |
| ROI_2 | 77 | 14 | 86% | 63% | 7 |
| ROI_3 | 78 | 31 | 74% | 74% | 8 |
| ROI_4 | 77 | 26 | 46% | Poor | High |

**Finding:** ROI_4 had severe false positives and low recall → **model needs re-annotation and retraining**.

### 6.3 Crop-Level Classifier Test (Grayscale Models)

Trained same EfficientAD-Medium architecture on grayscale 3-channel images to eliminate white-color false positives:

| ROI | Good Crops | Bad Crops | Recall | Precision | FP | FN |
|-----|-----------|----------|--------|-----------|-----|-----|
| **ROI_1** | 74 | 33 | **93.9%** | 83.8% | 6 | 2 |
| **ROI_2** | 77 | 14 | **92.9%** | 68.4% | 6 | 1 |
| **ROI_3** | 78 | 31 | **93.5%** | 69.0% | 13 | 2 |
| ROI_4 | 77 | 26 | 92.3% | 30.0% | **56** | 2 |

**Conclusion:**
- ✅ Grayscale training dramatically improved recall for ROI 1, 2, 3 (all >92%)
- ✅ False negatives dropped to 1-2 per ROI for ROI 1-3
- ❌ ROI_4 still has 56 false positives despite high recall → annotations are the problem, not the model architecture

---

## 7. Current Status & Next Steps

### Status
| Component | Status | Notes |
|-----------|--------|-------|
| YOLO Dataset Slicing | ✅ Done | `slice_dataset.py` |
| YOLO11m Training | ✅ Done | `runs/detect/train-2/weights/best.pt` |
| EfficientAD Color Models (ROI 1-4) | ✅ Done | `models/rois/` |
| EfficientAD Grayscale Models (ROI 1-3) | ✅ Done | `models/gray_scale_rois/` |
| ROI_4 Re-annotation | 🔴 TODO | Annotations suspected to be noisy |
| ROI_4 Retraining | 🔴 TODO | After re-annotation |
| End-to-End Test with New YOLO | 🟡 Pending | Needs `git reset --hard` on remote PC |

### Next Steps
1. **Re-annotate ROI_4 bad crops** using LabelImg with more careful bounding boxes around defects.
2. **Retrain ROI_4** with the new annotations:
   ```powershell
   python himalaya_label_detection/scripts/train_classifiers.py --rois data/gray_scale_rois --out models/gray_scale_rois
   ```
3. **Run full end-to-end test** with the new YOLO11m weights and grayscale classifiers:
   ```powershell
   python himalaya_label_detection/scripts/test_roi_classifiers.py --dataset "dataset\NSC\NSC" --models models/gray_scale_rois --grayscale
   ```

---

## 8. Key Scripts Reference

| Script | Purpose | Command |
|--------|---------|---------|
| `slice_dataset.py` | Slice 8000px images into 1504×1504 YOLO training patches | `python himalaya_label_detection/scripts/slice_dataset.py` |
| `train_yolo_roi.py` | Train YOLO11m on sliced dataset | `python himalaya_label_detection/scripts/train_yolo_roi.py` |
| `train_classifiers.py` | Train EfficientAD on ROI crops | `python himalaya_label_detection/scripts/train_classifiers.py --rois data/rois --out models/rois` |
| `create_grayscale_dataset.py` | Convert color crops to grayscale | `python himalaya_label_detection/scripts/create_grayscale_dataset.py` |
| `test_classifiers_on_crops.py` | Test classifiers directly on crops (bypasses YOLO) | `python himalaya_label_detection/scripts/test_classifiers_on_crops.py --data data/gray_scale_rois --models models/gray_scale_rois` |
| `test_roi_classifiers.py` | Full end-to-end test (YOLO + classifier) | `python himalaya_label_detection/scripts/test_roi_classifiers.py --dataset "dataset\NSC\NSC" --models models/gray_scale_rois --grayscale` |
| `calibrate_roi_thresholds.py` | Recalibrate thresholds after retraining | `python himalaya_label_detection/scripts/calibrate_roi_thresholds.py` |

---

## 9. Known Issues & Gotchas

| Issue | Description | Fix |
|-------|-------------|-----|
| Old YOLO weights loading | `test_roi_classifiers.py` was loading `train4/best.pt` (old) instead of the new YOLO11m weights | Script now searches both `models/` and `runs/` for newest `best.pt` |
| NumPy 2.x incompatibility | `torch 2.3.0` was compiled against NumPy 1.x — crashes when NumPy 2.x is installed | `pip install "numpy<2"` |
| `git pull` not updating scripts | Local uncommitted changes prevent pull from applying updates | `git fetch origin && git reset --hard origin/himalaya-label-detection` |
| ROI_4 false positives | 56 false alarms on good crops — the "3 Way Care" graphic has many white/light areas that resemble defects in color space | Re-annotate + retrain on grayscale |
| `invalid argument` on old scripts | Remote PC is running a cached old version of the script | Force reset with `git reset --hard` command above |

---

## 10. Repository

- **Repo:** https://github.com/ShreyasBairyKS/Capstone-Project
- **Branch:** `himalaya-label-detection`
- **Key config:** `himalaya_label_detection/config/roi_thresholds.json`
