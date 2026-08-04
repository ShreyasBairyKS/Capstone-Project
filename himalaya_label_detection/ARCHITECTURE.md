# Architecture — Himalaya NSC Label Defect Detection

## Problem Statement

We inspect printed labels on **Himalaya Winter Defense Moisturizing Cream (50 ml)** tubes.
The labels are scanned as flat unwrapped images — each image is roughly **1504 × 8000 pixels** in BMP format.
The goal is to automatically flag defects like torn print, smudges, missing text, ink blobs, or wrinkles.

We had **80 good images** and **42 defective images** to work with.

---

## Pipeline Overview

The label has four visually distinct regions. The problem is broken into two steps:

1. **Localize each region** — using template matching to find where each ROI sits in the full image
2. **Classify each region** — using an anomaly model to decide if it looks normal or defective

```
Full Image (1504 × 8000 px)
        │
        ▼
  Template Matching  ←── fixed anchor patches saved from reference images
        │
        ├── ROI_1 crop ──► EfficientAD ──► score + heatmap
        ├── ROI_2 crop ──► EfficientAD ──► score + heatmap
        ├── ROI_3 crop ──► EfficientAD ──► score + heatmap
        └── ROI_4 crop ──► EfficientAD ──► score + heatmap
                                 │
                      PASS / FAIL + annotated image
```

---

## Stage 1 — Region Localization (Template Matching)

### The 4 Regions (ROIs)

| ROI | Content |
|-----|---------|
| ROI_1 | Himalaya logo + product name |
| ROI_2 | Ingredient text (Jojoba Oil, Wheat Germ, Almond Oil) |
| ROI_3 | Address, regulatory info, MFG/EXP dates, barcode |
| ROI_4 | "3 Way Care" graphic with icons |

Since the label wraps around a cylindrical tube, **each ROI appears twice per image**. The matching logic always selects the complete (uncut) occurrence by checking which instance is fully within the image boundaries.

Template matching uses small reference patches (anchor crops) saved from known-good images. At runtime, OpenCV's `matchTemplate` finds the location of each patch in the new image, and the ROI is cropped from that position.

---

## Stage 2 — Anomaly Classification (EfficientAD-Medium)

### Why Anomaly Detection

We do not classify defect types — we only ask "does this region look normal?". Anomaly detection is the right fit because:
- The model trains **only on good images**, learning what normal looks like
- Any significant deviation at test time is flagged as a defect
- New defect types that were never seen during training will still be caught

### How EfficientAD Works

EfficientAD uses a **student-teacher** setup:

- A pre-trained **teacher** network extracts features from an image crop
- A **student** network is trained to mimic the teacher's output on good images
- An **autoencoder** is also trained to reconstruct good images
- On a defective crop, both the student and autoencoder fail on the defective region
- This produces a pixel-level **anomaly map** (heatmap) highlighting the defective area
- The mean of the anomaly map is the final anomaly score for that ROI

If the score exceeds a calibrated threshold → the ROI is flagged as defective.

### Why EfficientAD-Medium

| Variant | Speed | Accuracy |
|---------|-------|----------|
| EfficientAD-Small | Fastest | Lower |
| **EfficientAD-Medium** | Fast | **Our choice** |
| PatchCore | Slow (memory bank) | Highest |

PatchCore was considered but ruled out — it uses a feature memory bank that grows with dataset size, making it too slow for real-time use and unsuitable for edge deployment. EfficientAD-Medium gives a good balance and runs at ~50ms per crop on GPU.

### Input Size

Each ROI crop is resized to **256 × 256 pixels** before being passed to EfficientAD. This is the model's native input size and was kept fixed during all experiments.

### Threshold Calibration

After training, the model is scored on all known bad crops. The threshold is set at the **5th percentile of bad-image scores** — this guarantees at least 95% of real defects are caught. A small number of false alarms is accepted to avoid missing real defects.

Thresholds are stored per-ROI in `himalaya_label_detection/config/roi_thresholds.json`.

---

## Grayscale Training

Color crops were causing the model to flag large white/bright regions as anomalies on good images (false positives caused by color bias in the white label background).

We retrained using **grayscale crops** — converted to single-channel and then stacked to 3 channels so EfficientAD's input shape remains unchanged. Results on the test set:

| ROI | Color Recall | Grayscale Recall | FP (color) | FP (grayscale) |
|-----|-------------|-----------------|-----------|----------------|
| ROI_1 | ~79% | **93.9%** | 11 | 6 |
| ROI_2 | ~86% | **92.9%** | 7 | 6 |
| ROI_3 | ~74% | **93.5%** | 8 | 13 |

Grayscale training is now used for ROI_1, ROI_2, ROI_3. The model now focuses on texture and print intensity rather than color, which is more relevant for detecting print defects.

---

## Key Numbers

| Item | Detail |
|------|--------|
| Input image | ~1504 × 8000 px, BMP |
| ROI localization | Template matching (OpenCV) |
| Anomaly model | EfficientAD-Medium (anomalib 1.1.0) |
| Anomaly input size | 256 × 256 px |
| Training images (good) | 80 per ROI |
| Test set | 120 images (78 good, 42 bad) |
| Recall on crops (grayscale) | >92% for ROI_1, 2, 3 |
| Threshold strategy | 5th percentile of bad scores |
| GPU | NVIDIA RTX A5000 (24 GB) |
