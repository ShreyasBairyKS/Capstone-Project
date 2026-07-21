# Industrial Label Inspection System

## Optimized Hybrid Anomaly Detection Pipeline

**Goal:** Maximize recall (minimize missed defects) while maintaining
low inline latency for production.

------------------------------------------------------------------------

# 1. Target Defects

The proposed pipeline is designed to detect:

-   Black spots / ink blobs ✅
-   Tears / cuts in label ✅
-   Missing print ✅
-   Smudges / ink spread ✅
-   Scratches ✅
-   Wrinkles / folds ✅
-   Missing logo ✅
-   Wrong logo / artwork ✅
-   Missing barcode / QR damage ✅
-   Misalignment of repeated halves ✅
-   Color-independent texture defects ✅

Color-only defects (wrong ink shade) require RGB models instead of
grayscale.

------------------------------------------------------------------------

# 2. High Level Architecture

``` text
Industrial Camera
      │
      ▼
Marker Detection
      │
Perspective Correction
      │
Crop Label ROI
      │
Shared CNN Backbone
      │
ROI Feature Extraction
      │
 ┌────────────┬────────────┬────────────┐
 ▼            ▼            ▼
Top ROI   Center ROI   Bottom ROI
 │            │            │
Patch-based Anomaly Detection
 │            │            │
 └────────────┴────────────┘
        │
Decision Fusion
        │
PASS / FAIL
        │
If FAIL
        ▼
Heatmap Generation
        ▼
Crop Suspicious Area
        ▼
Segmentation
        ▼
Annotated Image + Operator Dashboard
```

------------------------------------------------------------------------

# 3. Pipeline Stages

## Stage 1 -- Image Acquisition

Use fixed industrial camera, constant lighting, fixed exposure and lens.

Algorithms: - Hardware trigger - LED ring/bar lighting - Polarizers
(optional)

Pros: - Removes lighting variance. - Improves downstream recall.

------------------------------------------------------------------------

## Stage 2 -- Marker Detection & Perspective Correction

Purpose: Normalize every label into the same coordinate system.

Algorithms: - ORB + Homography ⭐ Recommended - SIFT + Homography -
ArUco markers (if markers available)

Pros: - Deterministic - Very fast - Eliminates rotation and perspective
errors

Cons: - Requires reliable marker visibility.

------------------------------------------------------------------------

## Stage 3 -- ROI Extraction

Extract semantic regions instead of arbitrary crops.

Example:

1.  Top Logo
2.  Top Icons
3.  Center Information
4.  Bottom Logo
5.  Bottom Icons

Reason: Each region has different texture and defect characteristics.

------------------------------------------------------------------------

## Stage 4 -- Shared Backbone Feature Extraction

Instead of one CNN per ROI:

Run ONE backbone.

Recommended:

  ------------------------------------------------------------------------
  Backbone                         Pros                Cons
  -------------------------------- ------------------- -------------------
  EfficientNet-B0 ⭐               Excellent accuracy, Slightly slower
                                   lightweight         than MobileNet

  MobileNetV3                      Extremely fast      Slightly lower
                                                       accuracy

  ResNet18                         Stable and simple   Larger than
                                                       MobileNet
  ------------------------------------------------------------------------

Recommendation: EfficientNet-B0

Reason: Excellent feature quality with reasonable latency.

------------------------------------------------------------------------

## Stage 5 -- Patch-Based Anomaly Detection

Only normal images are required for training.

Recommended Models

### EfficientAD ⭐⭐⭐⭐⭐ (Best Overall)

Pros - State-of-the-art - Extremely fast - Excellent recall - Small
model - Heatmap output

Cons - Less community adoption than PatchCore

------------------------------------------------------------------------

### PatchCore ⭐⭐⭐⭐⭐

Pros - Industry standard - Outstanding recall - Heatmaps - No defective
samples required

Cons - Memory bank can become large

------------------------------------------------------------------------

### PaDiM ⭐⭐⭐⭐☆

Pros - Stable - Easy to train - Good with small datasets

Cons - Slightly lower recall than PatchCore

------------------------------------------------------------------------

### FastFlow ⭐⭐⭐⭐☆

Pros - Fast - Compact

Cons - Slightly more training complexity

------------------------------------------------------------------------

## Why Patch Models Detect Tears and Black Spots

Patch models learn ONLY normal appearance.

Therefore they detect:

-   Unknown black spots
-   Scratches
-   Missing ink
-   Torn corners
-   Paper texture changes
-   Wrinkles
-   Holes

without explicitly training every defect.

------------------------------------------------------------------------

## Stage 6 -- Decision Fusion

Inputs

-   Left-Right Similarity
-   Golden Master Similarity
-   ROI Anomaly Scores

Rule:

Reject if: - Left-right mismatch - Golden Master below threshold - Any
ROI anomaly score exceeds threshold

Thresholds are tuned using validation data to maximize recall while
keeping acceptable false positives.

------------------------------------------------------------------------

## Stage 7 -- Background Segmentation

Executed ONLY after rejection.

Purpose

-   Pixel localization
-   Operator visualization
-   Dataset generation
-   Failure analytics

Recommended Models

### SegFormer-B0 ⭐⭐⭐⭐⭐

Pros - Transformer accuracy - Lightweight - Excellent localization -
Generalizes well

Cons - Slightly more complex than U-Net

------------------------------------------------------------------------

### U-Net ⭐⭐⭐⭐☆

Pros - Very fast - Simple - Small memory footprint

Cons - Slightly weaker on complex textures

------------------------------------------------------------------------

### DeepLabV3+

Pros - Excellent segmentation

Cons - Higher latency

------------------------------------------------------------------------

Recommendation

SegFormer-B0

If GPU budget is very limited:

Use U-Net.

Do not use SAM or Mask R-CNN for inline industrial inspection due to
high latency.

------------------------------------------------------------------------

# 4. Left vs Right Verification

Algorithms

-   SSIM ⭐
-   ORB feature matching
-   Cosine similarity on embeddings

Detects

-   Missing duplicated print
-   Wrong repeated text
-   Misalignment

------------------------------------------------------------------------

# 5. Golden Master Verification

Recommended:

EfficientNet-B0 embeddings + Cosine Similarity

Alternative:

SSIM after alignment

------------------------------------------------------------------------

# 6. RGB vs Grayscale

Use Grayscale if:

-   Scratches
-   Tears
-   Black spots
-   Missing print
-   Wrinkles

Use RGB if:

-   Wrong ink color
-   Color fading
-   Brand color validation

Recommendation: Train anomaly model in grayscale unless color
correctness is a requirement.

------------------------------------------------------------------------

# 7. Performance Optimizations

-   Shared backbone for all ROIs
-   Semantic ROI extraction
-   Early exit after severe anomaly
-   Segmentation only on failed products
-   Segment only anomaly crop, not full image
-   Asynchronous visualization pipeline

------------------------------------------------------------------------

# 8. Expected Characteristics

  Metric           Expected
  ---------------- -------------------------------------
  Inline latency   Low (single backbone + ROI anomaly)
  Recall           Very High
  Precision        High after threshold tuning
  Missed defects   Very low
  Scalability      Excellent
  Maintenance      Low

------------------------------------------------------------------------

# 9. Final Recommendation

## Inline

-   ORB + Homography
-   Semantic ROI extraction
-   EfficientNet-B0 shared backbone
-   EfficientAD (preferred) or PatchCore
-   Left-vs-Right verification
-   Golden Master embedding comparison
-   Decision fusion

## Background

-   Heatmap
-   Crop anomaly
-   SegFormer-B0
-   Annotated image
-   Database and operator dashboard

This architecture maximizes recall for unknown and known defects while
keeping inference latency low by avoiding multiple independent CNNs and
executing expensive localization only after a product has already been
rejected.
