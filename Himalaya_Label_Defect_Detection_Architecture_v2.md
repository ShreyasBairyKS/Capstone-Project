# Himalaya Label Defect Detection System
## Architecture v2 — Functional ROI Design + Full Rationale + Build Plan

> **Product:** Himalaya Winter Defense Moisturizing Cream (50 ml tube)  
> **Goal:** Detect surface and print defects on the wrapped label at inline production speed  
> **Key design shift:** ROI naming changed from positional (Top / Center / Bottom) → **functional** (Logo / Barcode / Batch Code / Text / Graphics) for layout-change resilience

---

# Part 1 — Why Functional ROI Naming Matters

## The Problem with Positional ROIs

A positional scheme like `Top ROI`, `Center ROI`, `Bottom ROI` couples your detection logic to the **physical layout of a specific label version**. Himalaya labels are redesigned periodically — the barcode may move, the logo may shift, a new certification badge may appear. Every layout change requires re-mapping coordinates and retraining or re-calibrating all ROI detectors.

## Functional ROI Names Used in This System

| ROI Name         | What It Covers                                      | Defect Sensitivity |
|------------------|-----------------------------------------------------|--------------------|
| `ROI_LOGO`       | Himalaya "Since 1930" brand logo + tagline          | Logo missing, wrong logo, smear |
| `ROI_PRODUCT_NAME` | "Winter Defense Moisturizing Cream" title text    | Missing print, blur, smudge |
| `ROI_INGREDIENT_BAR` | "Jojoba Oil · Wheat Germ · Almond Oil" strip   | Text dropout, overprint |
| `ROI_GRAPHICS`   | Central decorative artwork (plant/flower imagery)   | Tear, ink bleed, wrong artwork |
| `ROI_CLAIMS`     | "3 Free From…" icon block                          | Icon missing, icon bleed |
| `ROI_BARCODE`    | Linear barcode strip                                | Barcode damage, smear, missing bars |
| `ROI_BATCH_CODE` | Batch number / MFG date / EXP date text            | Missing print, wrong font, smudge |
| `ROI_CERT`       | Certification marks (AYUSH / Natrue / etc.)         | Logo missing, wrong mark |
| `ROI_TEXT_BLOCK` | Full-label legal / description text panel          | Wrinkle, tear, missing text |

## Why This Is Resilient

Each ROI is **located by content detection**, not by fixed pixel coordinates:

- `ROI_LOGO` is found by detecting the Himalaya wordmark using template matching or a small classifier
- `ROI_BARCODE` is found by a barcode region detector (e.g., gradient orientation analysis)
- `ROI_BATCH_CODE` is found by detecting a small-font alphanumeric region near the barcode

If the label is redesigned and the barcode moves from bottom-right to top-left, `ROI_BARCODE` still finds it automatically. No pipeline rewrite is needed.

---

# Part 2 — Full Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    HIMALAYA LABEL INSPECTION                    │
└─────────────────────────────────────────────────────────────────┘

  Industrial Camera (fixed mount, constant exposure)
           │
           ▼
  ┌─────────────────────────┐
  │   Stage 1: Acquisition  │  ← Hardware trigger + LED ring lighting
  └─────────────────────────┘
           │
           ▼
  ┌───────────────────────────────────────┐
  │  Stage 2: Perspective Correction      │  ← ORB + Homography
  │  (align to golden master coordinate)  │
  └───────────────────────────────────────┘
           │
           ▼
  ┌───────────────────────────────────────┐
  │  Stage 3: Functional ROI Extraction   │  ← Content-based detection
  │  Logo | Barcode | Batch | Graphics    │
  │  Text | Claims | Cert | Ingredient    │
  └───────────────────────────────────────┘
           │
           ▼
  ┌───────────────────────────────────────┐
  │  Stage 4: Shared Backbone             │  ← EfficientNet-B0 (single forward pass)
  │  One CNN → feature maps for all ROIs  │
  └───────────────────────────────────────┘
           │
     ┌─────┴─────────────────────────────────────┐
     ▼                                           ▼
  ┌──────────────────────────┐     ┌─────────────────────────────┐
  │  Stage 5a:               │     │  Stage 5b:                  │
  │  Per-ROI Anomaly Score   │     │  Global Checks              │
  │  (EfficientAD / PatchCore│     │  · Left-Right SSIM match    │
  │   per functional region) │     │  · Golden Master similarity │
  └──────────────────────────┘     └─────────────────────────────┘
           │                                     │
           └──────────────┬──────────────────────┘
                          ▼
              ┌──────────────────────┐
              │  Stage 6:            │
              │  Decision Fusion     │  ← Weighted rule engine
              └──────────────────────┘
                          │
              ┌───────────┴──────────┐
              ▼                      ▼
           PASS                    FAIL
                                     │
                          ┌──────────▼───────────┐
                          │  Stage 7:             │
                          │  Anomaly Heatmap      │
                          │  → Crop suspect zone  │
                          │  → SegFormer-B0       │
                          │  → Pixel mask         │
                          └──────────┬────────────┘
                                     │
                          ┌──────────▼───────────┐
                          │  Stage 8:             │
                          │  Operator Dashboard   │
                          │  Annotated image +    │
                          │  ROI name + defect    │
                          │  category + confidence│
                          └───────────────────────┘
```

---

# Part 3 — Per-Stage Deep Rationale

---

## Stage 1 — Image Acquisition

**What:** Fixed industrial camera, constant aperture, exposure, and white balance. LED ring or coaxial bar lighting. Hardware trigger synchronized with conveyor.

**Why each element:**

| Element | Reason |
|---|---|
| Fixed camera mount | Eliminates frame-to-frame geometric variance, reducing the burden on perspective correction |
| Constant LED lighting | Himalaya labels have a blue gradient and white logo — reflective surfaces create false highlights under variable lighting; constant lighting makes anomaly thresholds stable |
| Hardware trigger | Software trigger introduces jitter; on a 200 ms/label line a 10 ms jitter causes blur and misalignment |
| Polarizer filter (optional) | Cream tubes are semi-glossy — a cross-polarizer removes specular reflection from the tube surface that would otherwise corrupt anomaly scores |

**Output:** 1 raw image per label, fixed resolution (recommended: 2048 × 1024 px for a 110 mm tube)

---

## Stage 2 — Perspective Correction with ORB + Homography

**What:** Detect 4+ keypoint correspondences between the incoming frame and a stored golden master image, then apply a projective transformation (homography) to warp the incoming frame into alignment.

**Why ORB over SIFT or AKAZE:**

| Method | Speed | Accuracy | Patent | Why chosen |
|---|---|---|---|---|
| **ORB** ✅ | Very Fast | Good | Free | Sufficient for rigid label; no GPU needed |
| SIFT | Slow | Excellent | Free (since 2020) | Overkill for a flat label |
| AKAZE | Medium | Very Good | Free | Good alternative if ORB keypoints are sparse |
| ArUco markers | Fastest | Perfect | Free | Only if markers can be printed on label edges — not always approved by brand |

**Why homography specifically:**  
A label on a tube exhibits perspective distortion and slight tilt as it moves under the camera. Homography (8-DOF planar transform) models exactly this: it corrects scale, rotation, shear, and perspective of a flat surface in one matrix multiplication. It does not model 3D bending of the cylindrical surface, but for a narrow camera strip the planarity approximation holds within 1–2 pixels.

**Critical for Himalaya labels specifically:**  
The label has rich texture (blue background, white logo text, plant graphic) providing dense keypoints. ORB will find 200–500 keypoints reliably. RANSAC inside `cv2.findHomography` discards outliers, making the step robust to partial occlusion or edge glare.

---

## Stage 3 — Functional ROI Extraction

**What:** After alignment, locate each semantic region by content rather than fixed coordinates.

**How each ROI is located:**

### `ROI_LOGO`
- Template match the Himalaya "H" logotype using normalized cross-correlation (NCC)  
- Or use a small binary classifier (MobileNetV3 head, 2-class) trained on logo-present vs. logo-absent patches  
- **Why:** The logo is the single highest-value brand element; a wrong or missing logo is a critical defect that must never be missed

### `ROI_BARCODE`
- Detect using gradient orientation variance (horizontal stripes = barcode)  
- Or use ZBar / pyzbar to simultaneously decode the barcode content  
- **Why two purposes:** (1) presence/quality check via anomaly score, (2) content check — wrong barcode string means wrong product variant on the line

### `ROI_BATCH_CODE`
- Small-font text region: detect using MSER (Maximally Stable Extremal Regions) or EAST text detector  
- **Why:** Batch codes are printed post-label-manufacture (jet ink on the label); they have different ink chemistry and are more prone to smear, missing print, and overprint than the base label

### `ROI_GRAPHICS`
- Locate the plant/flower artwork region: use color segmentation (blue→white gradient boundary) or a fixed relative offset from `ROI_LOGO` after alignment  
- **Why separate ROI:** The artwork region has continuous-tone gradients, making it the most sensitive region for tears, ink bleed, and color-shift defects — it requires a separately trained anomaly model

### `ROI_TEXT_BLOCK`
- Large text panels on the sides/back of the label  
- **Why:** Wrinkles and tears manifest primarily in text panels because they are the largest contiguous paper area; also missing-print defects are most visible here

### `ROI_CLAIMS` / `ROI_CERT` / `ROI_INGREDIENT_BAR`
- Located relative to `ROI_LOGO` after alignment using learned offsets  
- **Why:** These contain small logos and icons that must be pixel-accurate; icon smear or missing icon is a regulatory issue (e.g., missing AYUSH mark)

---

## Stage 4 — Shared Backbone (EfficientNet-B0)

**What:** Run a single EfficientNet-B0 forward pass over the full aligned label image and extract feature maps at multiple scales. Crop feature sub-regions corresponding to each functional ROI.

**Why shared rather than one CNN per ROI:**

| Approach | Forward passes | RAM | Latency | Accuracy |
|---|---|---|---|---|
| One CNN per ROI (8 ROIs) | 8× | 8× | ~400 ms | Redundant |
| **Shared backbone** ✅ | 1× | 1× | ~50 ms | Same or better |

**Why EfficientNet-B0 specifically:**

- **Compound scaling** — EfficientNet jointly scales depth, width, and resolution. B0 is the smallest member but retains structural efficiency not present in MobileNet or ResNet.
- **Feature density** — The intermediate feature maps of B0 at stride 8 (28×28 for a 224 px crop) provide dense spatial coverage ideal for patch-based anomaly detection downstream.
- **Transfer learning** — ImageNet weights provide powerful low-level texture features (edges, gradients, ink texture) out-of-the-box without label-specific pretraining.
- **Latency** — B0 runs at ~5 ms on a mid-range GPU (RTX 3060), well within inline constraints.

**Alternative:** MobileNetV3-Large if running on an embedded CPU (Jetson Nano / Raspberry Pi 5). Latency ~12 ms CPU-side with ONNX.

---

## Stage 5a — Per-ROI Anomaly Detection (EfficientAD / PatchCore)

**The Core Insight: Why Anomaly Detection Instead of Classification**

You do not need labeled defect images. Defects in production are rare and varied. A classifier trained on known defect types will always miss novel defect categories. Anomaly detection trains only on **good labels**, learning what "normal" looks like, and flags anything that deviates.

This is critical for Himalaya labels because:
- Tears can be any shape
- Black ink blobs can appear anywhere
- Wrinkles vary in direction and severity
- New defect types may appear as raw material suppliers change

---

### Recommended: EfficientAD

**What:** A student-teacher architecture. A large pretrained teacher network and a small student network are trained such that the student mimics the teacher on normal patches. At inference, regions where student and teacher disagree are anomalous.

**Why EfficientAD for this project:**

| Property | Why it matters here |
|---|---|
| State-of-the-art MVTec score | The MVTec benchmark is the standard for industrial surface inspection — the same domain as Himalaya labels |
| ~1 ms inference per ROI | Critical for inline production; 8 ROIs × 1 ms = 8 ms total anomaly scoring |
| Pixel-level heatmap output | Directly feeds Stage 7 without extra computation |
| No defective samples required | Himalaya QA team only needs to collect ~200–500 good labels |
| Small student model | Can be quantized to INT8 for edge deployment |

**How it works for a Himalaya label ROI:**  
The teacher (pretrained on ImageNet) produces rich feature descriptors for each patch. The student is trained to reproduce these descriptors only for normal patches. At inference on a torn or smeared patch, the student fails to reproduce the teacher's descriptor → high anomaly score → that patch is flagged.

---

### Alternative: PatchCore

**What:** Stores a memory bank of normal patch features (nearest-neighbor retrieval at inference). Anomaly score = distance from nearest normal patch in memory bank.

**Why PatchCore is a strong fallback:**

- Industry-standard benchmark results
- No student-teacher training — just build a coreset from good images
- Scales well if you have 1000+ good label images

**Tradeoff vs. EfficientAD:** Memory bank grows with dataset size and requires approximate nearest-neighbor (FAISS) for speed. EfficientAD has no memory bank — fixed inference cost regardless of dataset size.

---

### Per-ROI Model Strategy

Not all ROIs need the same sensitivity:

| ROI | Recommended Model | Sensitivity Setting |
|---|---|---|
| `ROI_LOGO` | PatchCore (exact memory match) | High threshold — any deviation is defect |
| `ROI_BARCODE` | ZBar decode + EfficientAD | Critical: both content + surface quality |
| `ROI_BATCH_CODE` | EfficientAD | High sensitivity — jet print is unreliable |
| `ROI_GRAPHICS` | EfficientAD | Medium — artwork has natural variation |
| `ROI_TEXT_BLOCK` | EfficientAD | Medium — wrinkles need catch |
| `ROI_CLAIMS` | PatchCore | High — icon presence is binary |
| `ROI_CERT` | PatchCore | High — regulatory compliance |
| `ROI_INGREDIENT_BAR` | EfficientAD | Medium |

---

## Stage 5b — Global Verification

### Left-Right Symmetry Check (SSIM)

**What:** The Himalaya cream label is printed as a double-wide strip and folded/cut to wrap the tube. In the camera image, two halves of the label often appear side-by-side. SSIM comparison between the halves catches:
- Missing left or right copy
- Misregistration of the two-up print
- One half torn while the other is intact

**Why SSIM:** Structural Similarity Index Measure compares luminance, contrast, and structure simultaneously. It is fast (vectorized), interpretable, and produces a spatial map that can be fed to the heatmap stage.

### Golden Master Similarity

**What:** Compare the full aligned label embedding (from EfficientNet-B0 global average pool) against a stored reference embedding using cosine similarity.

**Why cosine similarity on embeddings rather than SSIM on pixels:**  
Pixel-level SSIM is sensitive to minor printing density variation (acceptable process variation). Embedding cosine similarity operates in a compressed feature space where minor tonal variation is suppressed but structural differences (missing logo, wrong artwork) remain large. This gives lower false positives while preserving recall.

**Threshold guidance:** Cosine similarity < 0.90 → trigger FAIL flag. Tune on validation set.

---

## Stage 6 — Decision Fusion

**What:** Combine signals from all ROI anomaly scores, left-right check, and golden master check into a single PASS/FAIL decision.

**Why a rule-based fusion rather than a learned combiner:**

- With a trained combiner (e.g., logistic regression on scores), you need labeled defective samples. Anomaly detection was chosen to avoid this.
- Rule-based fusion is transparent, auditable, and adjustable by QA engineers without retraining.
- Regulatory environments (ISO 15378 for pharmaceutical packaging) often require explainable rejection logic.

**Fusion Logic:**

```
FAIL if ANY of the following:
  1. Any critical ROI anomaly score > CRITICAL_THRESHOLD
     (critical ROIs: Logo, Barcode, Cert, Batch Code)
  2. Any non-critical ROI anomaly score > GENERAL_THRESHOLD
  3. Left-Right SSIM < LR_THRESHOLD
  4. Golden Master cosine similarity < GM_THRESHOLD
  5. ZBar barcode decode FAILS or decoded string ≠ expected SKU

PASS otherwise
```

**Threshold tuning:** Use a validation set of 50–100 known-good and 50+ seeded-defect labels. Sweep thresholds and plot precision-recall curves. Set thresholds at F2 maximum (recall-weighted F-score, since missed defects are costlier than false positives in pharma packaging).

---

## Stage 7 — Heatmap Generation + Segmentation (Failure Path Only)

**Why run this only on failures:**  
Segmentation models (SegFormer-B0) add ~15–25 ms per inference. Running on every label doubles latency for no gain on good products. Gate behind FAIL decision.

### Heatmap

The EfficientAD student-teacher disagreement map is already a per-pixel anomaly score. Overlay on the label image with a colormap (jet or hot) to show QA operators exactly where the defect is.

### Defect Crop

Threshold the heatmap (e.g., top 5% anomaly pixels), compute bounding box, expand by 20 px margin, crop. Feed only this crop to the segmentation model — not the full image.

**Why crop first:** SegFormer processes 512×512 px crops at ~15 ms. A full 2048×1024 label image would take ~120 ms and most pixels would be background. Cropping to the defect region makes segmentation faster and more accurate.

### SegFormer-B0

**What:** A transformer-based semantic segmentation model.

**Why SegFormer over U-Net:**

| Model | Why preferred for this task |
|---|---|
| **SegFormer-B0** ✅ | Hierarchical transformer encoder captures long-range context — important for distinguishing a smear (localized) from a wrinkle (linear) |
| U-Net | Excellent for biomedical images; slightly weaker on label textures with complex print; use if GPU memory < 4 GB |
| DeepLabV3+ | Highest quality but 3× slower than SegFormer-B0; not suitable for inline use |
| SAM / Mask R-CNN | Designed for natural images; latency is too high; not recommended |

**Output:** Per-pixel defect mask → overlaid on original label image → sent to operator dashboard with ROI name, defect bounding box, and anomaly score.

---

## Stage 8 — Operator Dashboard

**What:** Annotated image per rejected label showing:
- Highlighted defect region(s)
- ROI name(s) where defect was found (functional names, e.g., "BARCODE damaged", "LOGO smear")
- Anomaly score and threshold
- ZBar decode result (if applicable)
- Timestamp, batch ID, camera ID

**Why functional ROI names on the dashboard:**  
Operators need to take corrective action. "Defect in ROI_BARCODE" is actionable (check inkjet printer head). "Defect in Bottom ROI" is not.

---

# Part 4 — RGB vs. Grayscale Decision

| Use Grayscale | Use RGB |
|---|---|
| Scratches, tears, wrinkles, missing print, black spots, smudges | Wrong ink color, color fading, brand blue color shift |
| Faster training and inference | Required for color certification validation |
| Lower memory bank size in PatchCore | Needed if QA spec includes colorimetric tolerance |

**Recommendation for Himalaya labels:**  
Train anomaly models in **grayscale** for structure-based defects (the majority).  
Add a separate **RGB color consistency check** (compare average hue/saturation in key regions like the blue background against the golden master) to catch ink-shade defects without complicating the anomaly pipeline.

---

# Part 5 — Build Plan

---

## Phase 0 — Environment & Data Setup (Week 1–2)

### 0.1 Hardware Setup
- [ ] Mount industrial camera above conveyor (fixed bracket, no vibration)
- [ ] Install LED ring lighting (5600K, constant current driver)
- [ ] Connect hardware trigger to PLC or encoder
- [ ] Verify constant exposure across 500-frame test capture

### 0.2 Software Environment
```bash
# Python 3.10 recommended (anomalib compatibility)
conda create -n himalaya_label python=3.10
conda activate himalaya_label

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install anomalib==1.0.0           # EfficientAD + PatchCore
pip install opencv-python-headless    # ORB, homography, SSIM
pip install scikit-image              # SSIM
pip install pyzbar                    # Barcode decode
pip install segmentation-models-pytorch  # SegFormer / U-Net
pip install transformers              # SegFormer weights
pip install fastapi uvicorn           # REST API for dashboard
pip install streamlit                 # Operator dashboard UI
pip install faiss-gpu                 # PatchCore memory bank
pip install albumentations            # Training augmentations
pip install mlflow                    # Experiment tracking
```

### 0.3 Data Collection
- Collect **500 good labels** minimum (covers normal print variation, slight tonal shift, normal wrapping)
- Collect **50 seeded defect labels** for threshold tuning only (not for anomaly training):
  - 10× black spot (ink blob)
  - 10× tear (scored with a scalpel on edge)
  - 10× smudge (finger-press before drying)
  - 10× wrinkle (manual fold)
  - 10× missing print (mask a region before printing)
- Store in:
  ```
  data/
    good/          ← 500+ images
    defect_seed/   ← 50 images with annotations
      images/
      masks/       ← binary pixel masks
  ```

---

## Phase 1 — Preprocessing Pipeline (Week 2–3)

### 1.1 Perspective Correction Module
**File:** `src/preprocessing/align.py`

```python
import cv2
import numpy as np

class LabelAligner:
    """
    Aligns an incoming label image to the golden master
    coordinate system using ORB keypoints + RANSAC homography.
    """
    def __init__(self, golden_master_path: str, max_features: int = 1000):
        self.orb = cv2.ORB_create(nfeatures=max_features)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.golden = cv2.imread(golden_master_path, cv2.IMREAD_GRAYSCALE)
        self.kp_gold, self.desc_gold = self.orb.detectAndCompute(self.golden, None)

    def align(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        kp, desc = self.orb.detectAndCompute(gray, None)
        matches = self.matcher.match(self.desc_gold, desc)
        matches = sorted(matches, key=lambda m: m.distance)[:200]

        if len(matches) < 10:
            raise ValueError("Insufficient keypoints — check lighting or lens focus")

        src_pts = np.float32([self.kp_gold[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

        h, w = self.golden.shape[:2]
        return cv2.warpPerspective(image, H, (w, h))
```

**Why RANSAC threshold 5.0 px:** Matches the expected sub-5-pixel alignment tolerance for a 2048 px image. Outlier matches (from reflections or partial occlusion) are rejected.

### 1.2 ROI Extraction Module
**File:** `src/preprocessing/roi_extractor.py`

```python
from dataclasses import dataclass
from typing import Dict
import numpy as np
import cv2

@dataclass
class ROIConfig:
    name: str
    # Offset from aligned golden master (x, y, w, h) in pixels
    # These are set once from golden master annotation, then used forever
    x: int; y: int; w: int; h: int
    critical: bool = False

# Example config for a 2048×1024 label image
# Actual values must be measured from your golden master
ROI_CONFIGS = {
    "ROI_LOGO":          ROIConfig("ROI_LOGO",          120, 80,  400, 200, critical=True),
    "ROI_PRODUCT_NAME":  ROIConfig("ROI_PRODUCT_NAME",  100, 290, 500, 120),
    "ROI_INGREDIENT_BAR":ROIConfig("ROI_INGREDIENT_BAR", 50, 60,  600,  60),
    "ROI_GRAPHICS":      ROIConfig("ROI_GRAPHICS",       520,100, 400, 350),
    "ROI_CLAIMS":        ROIConfig("ROI_CLAIMS",          50, 420, 580, 160),
    "ROI_BARCODE":       ROIConfig("ROI_BARCODE",        900, 700, 300, 120, critical=True),
    "ROI_BATCH_CODE":    ROIConfig("ROI_BATCH_CODE",     900, 820, 300,  60, critical=True),
    "ROI_CERT":          ROIConfig("ROI_CERT",           840, 400, 150, 120, critical=True),
    "ROI_TEXT_BLOCK":    ROIConfig("ROI_TEXT_BLOCK",    1100,  50, 900, 980),
}

class ROIExtractor:
    def __init__(self, configs: Dict[str, ROIConfig] = ROI_CONFIGS):
        self.configs = configs

    def extract(self, aligned_image: np.ndarray) -> Dict[str, np.ndarray]:
        crops = {}
        for name, cfg in self.configs.items():
            crop = aligned_image[cfg.y:cfg.y+cfg.h, cfg.x:cfg.x+cfg.w]
            crops[name] = crop
        return crops
```

> **Note:** After initial setup, add a content-verification step per ROI (template match / text detection) to confirm the correct region was extracted. This enables layout-change detection.

---

## Phase 2 — Anomaly Model Training (Week 3–5)

### 2.1 Train EfficientAD per ROI

Use the `anomalib` library which ships with EfficientAD and PatchCore implementations.

**File:** `src/training/train_roi.py`

```python
from anomalib.data import Folder
from anomalib.models import EfficientAd
from anomalib.engine import Engine

def train_roi(roi_name: str, train_dir: str, output_dir: str):
    """
    Train an EfficientAD model on good-only patches from one ROI.
    No defective images required.
    """
    datamodule = Folder(
        name=roi_name,
        root=train_dir,          # contains only 'good/' subfolder
        normal_dir="good",
        image_size=(256, 256),
    )
    model = EfficientAd()
    engine = Engine(max_epochs=100, accelerator="gpu")
    engine.fit(model=model, datamodule=datamodule)
    engine.save_model(output_dir=f"{output_dir}/{roi_name}")
    print(f"[{roi_name}] Training complete → {output_dir}/{roi_name}")
```

**Training data layout:**
```
data/rois/
  ROI_LOGO/
    good/       ← 500 logo crops (good labels)
  ROI_BARCODE/
    good/       ← 500 barcode crops
  ...
```

**Why train separately per ROI:**  
Each ROI has a distinct visual distribution. `ROI_BARCODE` is thin black lines on white. `ROI_GRAPHICS` is smooth blue-to-white gradients. A single model trained on mixed ROIs would have poor recall in any one region. Separate models → each model has a tight normal distribution → higher anomaly signal-to-noise ratio.

### 2.2 Threshold Calibration

```python
# After training, score the 50 seeded defect images
# Plot histogram of anomaly scores for good vs. defect
# Set threshold at point where recall ≥ 0.99 (miss rate < 1%)

from anomalib.metrics import AUROC, F1Score
# Use anomalib's built-in threshold optimization
engine.test(model=model, datamodule=datamodule_with_defects)
```

---

## Phase 3 — Global Verification (Week 4–5)

### 3.1 Left-Right SSIM Check
**File:** `src/verification/lr_check.py`

```python
from skimage.metrics import structural_similarity as ssim
import numpy as np

def check_left_right(aligned_image: np.ndarray, threshold: float = 0.85) -> dict:
    h, w = aligned_image.shape[:2]
    left  = aligned_image[:, :w//2]
    right = aligned_image[:, w//2:]
    right_flipped = np.fliplr(right)
    score, diff_map = ssim(left, right_flipped, full=True, data_range=255)
    return {"score": score, "pass": score >= threshold, "diff_map": diff_map}
```

### 3.2 Golden Master Cosine Similarity
**File:** `src/verification/gm_check.py`

```python
import torch
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as T

class GoldenMasterChecker:
    def __init__(self, golden_master_image_path: str):
        self.model = models.efficientnet_b0(pretrained=True)
        self.model.classifier = torch.nn.Identity()  # Remove head, use embedding
        self.model.eval()
        self.transform = T.Compose([T.Resize((224,224)), T.ToTensor(),
                                    T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
        from PIL import Image
        gm = Image.open(golden_master_image_path).convert("RGB")
        with torch.no_grad():
            self.gm_embedding = self.model(self.transform(gm).unsqueeze(0))

    def score(self, image_tensor: torch.Tensor, threshold: float = 0.90) -> dict:
        with torch.no_grad():
            emb = self.model(image_tensor.unsqueeze(0))
        sim = F.cosine_similarity(self.gm_embedding, emb).item()
        return {"similarity": sim, "pass": sim >= threshold}
```

---

## Phase 4 — Decision Fusion (Week 5)

**File:** `src/fusion/decision.py`

```python
from dataclasses import dataclass
from typing import Dict

CRITICAL_ROIS = {"ROI_LOGO", "ROI_BARCODE", "ROI_BATCH_CODE", "ROI_CERT"}

@dataclass
class FusionConfig:
    critical_threshold: float = 0.5     # Anomaly score 0–1
    general_threshold:  float = 0.6
    lr_threshold:       float = 0.85
    gm_threshold:       float = 0.90

def fuse(roi_scores: Dict[str, float],
         lr_score: float,
         gm_score: float,
         barcode_valid: bool,
         cfg: FusionConfig = FusionConfig()) -> dict:

    reasons = []

    for roi, score in roi_scores.items():
        thr = cfg.critical_threshold if roi in CRITICAL_ROIS else cfg.general_threshold
        if score > thr:
            reasons.append(f"{roi} anomaly score {score:.3f} > {thr}")

    if lr_score < cfg.lr_threshold:
        reasons.append(f"L-R SSIM {lr_score:.3f} < {cfg.lr_threshold}")

    if gm_score < cfg.gm_threshold:
        reasons.append(f"Golden Master similarity {gm_score:.3f} < {cfg.gm_threshold}")

    if not barcode_valid:
        reasons.append("Barcode decode failed or SKU mismatch")

    return {"pass": len(reasons) == 0, "reasons": reasons}
```

---

## Phase 5 — Post-Rejection Analysis (Week 6)

### 5.1 Heatmap Overlay
```python
import cv2
import numpy as np

def overlay_heatmap(image: np.ndarray, anomaly_map: np.ndarray,
                    alpha: float = 0.5) -> np.ndarray:
    norm_map = cv2.normalize(anomaly_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    colormap = cv2.applyColorMap(norm_map, cv2.COLORMAP_JET)
    return cv2.addWeighted(image, 1-alpha, colormap, alpha, 0)
```

### 5.2 SegFormer-B0 Fine-Tuning (Optional but recommended)
- Collect 100–200 annotated defect crops (from seeded defects + early production rejects)
- Fine-tune SegFormer-B0 on defect segmentation
- Classes: `background`, `tear`, `black_spot`, `smudge`, `wrinkle`, `missing_print`
- Use `segmentation_models_pytorch` or HuggingFace `transformers` SegFormer implementation

```bash
# Fine-tune SegFormer-B0 on defect crops
python src/segmentation/train_segformer.py \
  --data_dir data/defect_crops \
  --num_classes 6 \
  --epochs 50 \
  --backbone nvidia/mit-b0
```

---

## Phase 6 — Integration & Deployment (Week 7–8)

### 6.1 Inference Pipeline Assembly
**File:** `src/pipeline.py`

```python
class HimalayaLabelInspector:
    def __init__(self, config_path: str):
        self.aligner        = LabelAligner(...)
        self.roi_extractor  = ROIExtractor(...)
        self.anomaly_models = {name: load_model(name) for name in ROI_CONFIGS}
        self.gm_checker     = GoldenMasterChecker(...)
        self.segformer      = load_segformer(...)

    def inspect(self, raw_image: np.ndarray) -> dict:
        aligned     = self.aligner.align(raw_image)
        roi_crops   = self.roi_extractor.extract(aligned)
        
        roi_scores  = {name: self.anomaly_models[name].score(crop)
                       for name, crop in roi_crops.items()}
        
        lr_result   = check_left_right(aligned)
        gm_result   = self.gm_checker.score(to_tensor(aligned))
        bc_result   = decode_barcode(roi_crops["ROI_BARCODE"])
        
        decision    = fuse(roi_scores, lr_result["score"],
                          gm_result["similarity"], bc_result["valid"])
        
        if not decision["pass"]:
            heatmap   = generate_heatmap(roi_scores, roi_crops)
            seg_mask  = self.segformer.segment(heatmap.crop)
            decision["heatmap"]  = heatmap
            decision["seg_mask"] = seg_mask
        
        return decision
```

### 6.2 Latency Budget

| Stage | Estimated Time |
|---|---|
| Image acquisition (hardware) | 5 ms |
| ORB alignment | 8 ms |
| ROI extraction | 1 ms |
| EfficientNet-B0 backbone | 5 ms |
| EfficientAD × 9 ROIs | 9 ms |
| L-R + GM check | 3 ms |
| Decision fusion | <1 ms |
| **Total (PASS path)** | **~30 ms** |
| SegFormer (FAIL path only) | +20 ms |
| **Total (FAIL path)** | **~50 ms** |

Supports 20 labels/second throughput on a single RTX 3060 GPU.

### 6.3 Streamlit Operator Dashboard
```bash
streamlit run src/dashboard/app.py
```
- Shows last N inspected labels
- Highlights rejected labels with annotated heatmap
- Displays ROI name, defect type, confidence
- Exportable rejection log (CSV / JSON) for QA traceability

### 6.4 ONNX Export for Edge Deployment
```python
# Export EfficientNet-B0 backbone + all EfficientAD heads to ONNX
torch.onnx.export(model, dummy_input, "models/efficientad_roi_logo.onnx",
                  opset_version=17, dynamic_axes={"input": {0: "batch"}})
# Quantize to INT8 with TensorRT or OpenVINO for Jetson / IPC deployment
```

---

## Phase 7 — Validation & Go-Live (Week 9–10)

### Acceptance Criteria

| Metric | Target |
|---|---|
| Recall (missed defect rate) | ≥ 99% |
| False positive rate | ≤ 5% |
| Inline latency (P95) | ≤ 50 ms |
| Throughput | ≥ 15 labels/second |
| Unseen defect detection | ≥ 90% (anomaly generalization) |

### Validation Protocol
1. Run 1000 known-good labels → measure false positive rate
2. Run 50-label seeded defect set (each defect type × 10) → measure per-class recall
3. Run 20 novel defect labels (not seen during threshold tuning) → measure generalization
4. Stress test at max production speed → measure latency distribution

---

# Part 6 — Summary: Key Design Decisions

| Decision | What | Why |
|---|---|---|
| Functional ROI naming | `ROI_BARCODE` not `Bottom ROI` | Layout-change resilience; operator-readable failure reports |
| ORB + Homography | Perspective correction | Fast, patent-free, sufficient for flat label alignment |
| Shared EfficientNet-B0 | One backbone for all ROIs | 8× less compute vs. per-ROI CNNs; consistent feature space |
| EfficientAD | Anomaly detection | No defect labels needed; detects unknown defects; heatmap output |
| Separate model per ROI | Different model per functional region | Each ROI has a distinct texture distribution; mixed training degrades recall |
| Grayscale anomaly + RGB color check | Two separate checks | Structural defects (grayscale) + ink-shade defects (RGB) require different features |
| SegFormer-B0 on FAIL path only | Post-rejection segmentation | Saves ~20 ms per label on the majority (good) path |
| Rule-based decision fusion | Explicit threshold rules | Auditable, adjustable without retraining, explainable under ISO 15378 |
| ZBar barcode decode | Content verification | Catches wrong SKU printed on label — pure anomaly models cannot verify text content |
| Cosine similarity for Golden Master | Global structural check | Suppresses acceptable printing variation while catching gross structural errors |
