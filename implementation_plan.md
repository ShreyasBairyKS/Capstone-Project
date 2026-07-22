# Himalaya NSC Defect Detection — Sprint Plan v3
## 3-Person Team | A500 GPU | ROI Crops Already Ready

---

## Updated Reality: What You Have Now

| Item | Status |
|------|--------|
| Raw NSC images (bad) | ✅ 42 full BMP images |
| Raw NSC images (good) | ✅ 78 full BMP images |
| **4 ROI folders (grayscale crops)** | ✅ **Already on remote PC — READY** |
| Annotations | ❌ None yet |
| Data prep pipeline | ✅ **SKIP — already done** |
| Previous failure reason | Translational variance (fixed by cropping ROIs) |

> [!IMPORTANT]
> Since you already have 4 ROI folders of cropped grayscale images, **Person A's old Task A1 (alignment) and Task A3 (data prep) are eliminated**. The ROI cropping itself already solved the translational variance problem for training. For inference on new images you still need alignment — but that is now Person A's only engineering focus.

---

## What the 4 ROI Folders Replace

The pre-cropped ROI folders already represent the output of:
- Stage 2 (Perspective Correction)
- Stage 3 (ROI Extraction)

So your pipeline work starts at **Stage 4 (Anomaly Model Training)** — which is a major head start.

---

## Revised Pipeline (What Still Needs Building)

```
┌─────────────────────────────────────────────────────┐
│  DONE (ROI crops exist):                            │
│  Stage 1: Acquisition ✅                            │
│  Stage 2: Alignment / unwrap ✅                     │
│  Stage 3: ROI Extraction ✅  (4 folders ready)      │
└─────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────┐   ← Person B (training)
│  Stage 4: EfficientAD Training per ROI folder       │
│  • Train on good crops only (grayscale, 256×256)    │
│  • 4 models, one per ROI folder                     │
└─────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────┐   ← Person B + C together
│  Stage 5: Threshold Calibration                     │
│  • Score good crops → baseline distribution         │
│  • Score bad ROI crops → anomaly distribution       │
│  • Set per-ROI threshold for PASS/FAIL              │
└─────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────┐   ← Person A (inference)
│  Stage 6: Inference Pipeline                        │
│  • Load trained models                              │
│  • Run new image → align → crop ROIs → score        │
│  • Heatmap → Bounding Box extraction                │
│  • Output: PASS/FAIL + ROI name + bbox + score      │
└─────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────┐   ← Person A
│  Stage 7: Simple Dashboard                          │
│  • Show image with bounding boxes drawn             │
│  • ROI name, anomaly score, PASS/FAIL               │
└─────────────────────────────────────────────────────┘
```

---

## Dependency Map (Parallel vs Sequential)

```
PARALLEL (can start immediately, no dependencies):
  ├── Person B: Setup env + start EfficientAD training    [Day 1]
  ├── Person C: Annotate bad ROI crops                    [Day 1]
  └── Person A: Build inference pipeline skeleton         [Day 1]

SEQUENTIAL (must wait):
  Person B threshold calibration
    └── WAITS FOR: Person C to finish annotating bad crops
                   (needs ground truth to know where anomaly score cutoff should be)

  Person A full inference pipeline
    └── WAITS FOR: Person B to finish training + thresholds
                   (needs trained model weights + threshold values)

  Final evaluation
    └── WAITS FOR: everything above
```

### Visual Timeline

```
         Day 1          Day 2          Day 3          Day 4
Person A │▓▓▓ Pipeline  │▓▓▓ Pipeline  │▓▓ Integrate  │▓ Test+Fix  │
         │  skeleton    │  + bbox code │  models      │            │
         │              │              │              │            │
Person B │▓▓▓ Setup +   │▓▓▓▓▓▓▓▓▓▓▓▓ │▓▓▓ Threshold │▓ Eval+    │
         │  training    │  Training    │  calibration │  metrics  │
         │  starts      │  all 4 ROIs  │  (needs C)   │            │
         │              │              │              │            │
Person C │▓▓▓▓▓▓▓▓▓▓▓▓▓│▓▓▓ Annotate  │▓ Done +      │▓ Manual   │
         │  Annotate    │  remaining   │  validate    │  review   │
         │  bad crops   │  bad crops   │  results     │            │
         ├──────────────┼──────────────┼──────────────┼────────────┤
         │              │         ↑ Handoff: C→B      │            │
         │              │         Annotations ready   │            │
         │              │                      ↑ Handoff: B→A      │
         │              │                      Weights+thresholds  │
```

---

# 🔴 Person A — Inference Pipeline + Bounding Box Output

**Starts:** Day 1 (in parallel with training)
**Total time:** ~3-4 days

Person A's job is to build the complete inference pipeline that takes a **new unseen image**, runs it through the trained models, and outputs **bounding boxes around detected anomalies**.

---

## Task A1: Build Inference Pipeline Skeleton (Day 1-2)

Build this before models are trained — use dummy/random outputs so the pipeline shape is testable end-to-end.

**File:** `src/pipeline.py`

```python
import cv2
import numpy as np
import torch
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple

@dataclass
class AnomalyResult:
    roi_name: str
    anomaly_score: float          # overall score for this ROI (0.0–1.0)
    anomaly_map: np.ndarray       # per-pixel heatmap (H x W float32)
    bounding_boxes: List[dict]    # list of {x, y, w, h, score} dicts
    is_anomalous: bool
    threshold_used: float

@dataclass
class InspectionResult:
    image_path: str
    overall_pass: bool
    roi_results: Dict[str, AnomalyResult]
    annotated_image: np.ndarray   # original image with bboxes drawn

class HimalayaInspector:
    def __init__(self, model_dir: str, threshold_config: str):
        self.models = self._load_models(model_dir)
        self.thresholds = self._load_thresholds(threshold_config)
        # ROI crop coordinates — must match how the remote PC cropped the 4 folders
        # Person C should confirm these exact pixel offsets
        self.roi_coords = {
            "ROI_LOGO":        (x1, y1, x2, y2),   # fill from roi_config.json
            "ROI_INGREDIENT":  (x1, y1, x2, y2),
            "ROI_GRAPHICS":    (x1, y1, x2, y2),
            "ROI_BARCODE":     (x1, y1, x2, y2),
        }

    def inspect(self, image_path: str) -> InspectionResult:
        raw = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        aligned = self._align(raw)                        # ORB homography
        roi_crops = self._extract_rois(aligned)           # 4 crops
        roi_results = {}
        for roi_name, crop in roi_crops.items():
            result = self._score_roi(roi_name, crop)
            roi_results[roi_name] = result
        overall_pass = all(r.is_anomalous == False for r in roi_results.values())
        annotated = self._draw_results(aligned, roi_results)
        return InspectionResult(image_path, overall_pass, roi_results, annotated)
```

---

## Task A2: Bounding Box Extraction from Anomaly Heatmap (Day 1-2)

This is the core output requirement. EfficientAD gives a per-pixel anomaly score map. Convert it to bounding boxes using connected components.

**File:** `src/postprocessing/bbox.py`

```python
import cv2
import numpy as np
from typing import List, Dict

def heatmap_to_bboxes(
    anomaly_map: np.ndarray,          # float32 array, shape (H, W), values 0.0–1.0
    threshold: float = 0.5,           # pixels above this → anomalous
    min_area_px: int = 50,            # ignore tiny noise blobs
    margin_px: int = 10,              # expand bbox slightly for visibility
) -> List[Dict]:
    """
    Convert a per-pixel anomaly score map into bounding boxes.
    
    Returns:
        List of dicts: [{x, y, w, h, score, area}]
        where score = mean anomaly value inside the bbox
    """
    # Step 1: Normalize to 0–255 uint8
    norm = cv2.normalize(anomaly_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Step 2: Binary threshold
    thresh_val = int(threshold * 255)
    _, binary = cv2.threshold(norm, thresh_val, 255, cv2.THRESH_BINARY)
    
    # Step 3: Morphological close — merge nearby blobs into one region
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # Step 4: Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )
    
    bboxes = []
    h_map, w_map = anomaly_map.shape
    
    for i in range(1, num_labels):  # skip label 0 = background
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area_px:
            continue  # ignore noise
        
        x = max(0, stats[i, cv2.CC_STAT_LEFT] - margin_px)
        y = max(0, stats[i, cv2.CC_STAT_TOP] - margin_px)
        w = min(w_map - x, stats[i, cv2.CC_STAT_WIDTH] + 2 * margin_px)
        h = min(h_map - y, stats[i, cv2.CC_STAT_HEIGHT] + 2 * margin_px)
        
        # Mean anomaly score inside the component region
        component_mask = (labels == i).astype(np.uint8)
        mean_score = float(np.mean(anomaly_map[component_mask == 1]))
        
        bboxes.append({
            "x": int(x), "y": int(y),
            "w": int(w), "h": int(h),
            "score": round(mean_score, 4),
            "area": int(area),
            "centroid": (float(centroids[i][0]), float(centroids[i][1]))
        })
    
    # Sort by score descending — highest anomaly first
    bboxes.sort(key=lambda b: b["score"], reverse=True)
    return bboxes


def map_bbox_to_original(
    bbox: Dict,
    roi_offset: Tuple[int, int],      # (x_offset, y_offset) of ROI in full image
    roi_crop_size: Tuple[int, int],    # (crop_w, crop_h) after resize
    original_roi_size: Tuple[int, int] # (orig_w, orig_h) before resize to model input
) -> Dict:
    """
    Map a bounding box from the model's input space (e.g. 256×256)
    back to the full aligned image coordinate space.
    """
    scale_x = original_roi_size[0] / roi_crop_size[0]
    scale_y = original_roi_size[1] / roi_crop_size[1]
    
    return {
        "x": int(bbox["x"] * scale_x) + roi_offset[0],
        "y": int(bbox["y"] * scale_y) + roi_offset[1],
        "w": int(bbox["w"] * scale_x),
        "h": int(bbox["h"] * scale_y),
        "score": bbox["score"],
        "area": bbox["area"]
    }
```

---

## Task A3: Result Visualization (Day 2-3)

Draw bounding boxes and heatmap overlay on the original image.

**File:** `src/postprocessing/visualize.py`

```python
import cv2
import numpy as np

# Color scheme per ROI (BGR format)
ROI_COLORS = {
    "ROI_LOGO":       (0, 0, 255),    # Red — critical
    "ROI_BARCODE":    (0, 0, 255),    # Red — critical
    "ROI_INGREDIENT": (0, 165, 255),  # Orange — important
    "ROI_GRAPHICS":   (0, 255, 255),  # Yellow — medium
}

def draw_inspection_result(
    image: np.ndarray,
    roi_results: dict,
    roi_coords: dict,
) -> np.ndarray:
    """
    Draw on the full aligned image:
    - Green ROI border if PASS, Red ROI border if FAIL
    - Bounding boxes around each detected anomaly (with score label)
    - Semi-transparent heatmap overlay on each ROI
    """
    vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image.copy()
    
    for roi_name, result in roi_results.items():
        x1, y1, x2, y2 = roi_coords[roi_name]
        color = ROI_COLORS.get(roi_name, (255, 255, 0))
        
        # Draw ROI boundary
        border_color = (0, 0, 255) if result.is_anomalous else (0, 255, 0)
        cv2.rectangle(vis, (x1, y1), (x2, y2), border_color, 2)
        cv2.putText(vis, roi_name, (x1 + 5, y1 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, border_color, 1)
        
        # Overlay heatmap on ROI region
        roi_h, roi_w = y2 - y1, x2 - x1
        heatmap_resized = cv2.resize(result.anomaly_map, (roi_w, roi_h))
        heatmap_norm = cv2.normalize(heatmap_resized, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)
        roi_region = vis[y1:y2, x1:x2]
        vis[y1:y2, x1:x2] = cv2.addWeighted(roi_region, 0.6, heatmap_color, 0.4, 0)
        
        # Draw anomaly bounding boxes (mapped back to full image coords)
        for bbox in result.bounding_boxes:
            bx = x1 + bbox["x"]
            by = y1 + bbox["y"]
            bx2 = bx + bbox["w"]
            by2 = by + bbox["h"]
            cv2.rectangle(vis, (bx, by), (bx2, by2), color, 2)
            label = f"{result.roi_name}: {bbox['score']:.2f}"
            cv2.putText(vis, label, (bx, by - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    
    # Overall verdict banner
    verdict = "PASS" if all(not r.is_anomalous for r in roi_results.values()) else "FAIL"
    banner_color = (0, 200, 0) if verdict == "PASS" else (0, 0, 220)
    cv2.rectangle(vis, (0, 0), (200, 40), banner_color, -1)
    cv2.putText(vis, verdict, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    
    return vis
```

---

## Task A4: Integrate + Test (Day 3-4)
Once Person B hands over model weights + thresholds:
- Wire `HimalayaInspector` with real loaded models
- Run 10 good + 10 bad images, visually confirm bounding boxes appear only on bad
- Save output images to `outputs/` for review

**Deliverables from Person A:**
- [ ] `src/pipeline.py` — end-to-end inspector class
- [ ] `src/postprocessing/bbox.py` — heatmap → bounding box conversion
- [ ] `src/postprocessing/visualize.py` — annotated output image
- [ ] `scripts/run_inference.py` — run on a folder of images, save results
- [ ] Sample output images showing bboxes on defective regions

---

# 🟡 Person B — EfficientAD Training + Threshold Calibration

**Starts:** Day 1 (fully parallel, ROI crops already ready)
**Total time:** ~3 days

---

## Task B1: Environment Setup (Day 1 — first 2 hours)

```bash
# On remote PC (has A500 GPU)
conda create -n himalaya python=3.10
conda activate himalaya
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install anomalib==1.1.0
pip install opencv-python scikit-image pillow albumentations
```

> [!NOTE]
> Verify anomalib version: `python -c "import anomalib; print(anomalib.__version__)"`. Use 1.x API only — 0.x API is completely different.

---

## Task B2: Confirm ROI Folder Structure (Day 1 — 1 hour)

Before training, verify your 4 folders match anomalib's expected layout:

```
data/rois/
  ROI_LOGO/
    good/        ← grayscale crops from good images (training data)
    bad/         ← grayscale crops from bad images (for threshold tuning ONLY)
  ROI_INGREDIENT/
    good/
    bad/
  ROI_GRAPHICS/
    good/
    bad/
  ROI_BARCODE/
    good/
    bad/
```

> [!WARNING]
> If the bad/ subfolder doesn't exist yet, create it and move bad image ROI crops there. Person C will be annotating these crops — make sure the filenames match between the crops and what Person C annotates.

Check image sizes: `python -c "from PIL import Image; import os; imgs = os.listdir('data/rois/ROI_LOGO/good'); img = Image.open(f'data/rois/ROI_LOGO/good/{imgs[0]}'); print(img.size, img.mode)"`

Expected: e.g. `(256, 256) L` (grayscale) or similar fixed size.

---

## Task B3: Train EfficientAD on All 4 ROI Folders (Day 1-2)

```python
# src/training/train_all_rois.py
from anomalib.data import Folder
from anomalib.models import EfficientAd
from anomalib.engine import Engine
import os

ROI_NAMES = ["ROI_LOGO", "ROI_INGREDIENT", "ROI_GRAPHICS", "ROI_BARCODE"]
DATA_ROOT = "data/rois"
OUTPUT_DIR = "models"

for roi_name in ROI_NAMES:
    print(f"\n{'='*50}\nTraining: {roi_name}\n{'='*50}")
    
    datamodule = Folder(
        name=roi_name,
        root=os.path.join(DATA_ROOT, roi_name),
        normal_dir="good",
        # Do NOT pass abnormal_dir during training — anomaly detection is unsupervised
        image_size=(256, 256),
        train_batch_size=16,
        num_workers=4,
    )
    
    model = EfficientAd(
        model_size="small",    # "small" or "medium" — start with small (faster)
    )
    
    engine = Engine(
        max_epochs=100,
        accelerator="gpu",
        devices=1,
        default_root_dir=os.path.join(OUTPUT_DIR, roi_name),
    )
    
    engine.fit(model=model, datamodule=datamodule)
    print(f"[{roi_name}] Training done → {OUTPUT_DIR}/{roi_name}")
```

**A500 GPU estimate:** Each ROI trains in ~8–15 min. All 4 = ~1 hour total. Run overnight if needed.

> [!TIP]
> Train all 4 in sequence (one script) so you can walk away. Add `torch.cuda.empty_cache()` between each ROI to free GPU memory.

---

## Task B4: Threshold Calibration (Day 3 — after Person C finishes annotation)

This is **sequential** — waits for Person C to finish annotating bad ROI crops.

```python
# scripts/calibrate_thresholds.py
import numpy as np
import json
from anomalib.engine import Engine

thresholds = {}

for roi_name in ROI_NAMES:
    # Load trained model
    engine = Engine(accelerator="gpu", devices=1)
    model = EfficientAd.load_from_checkpoint(f"models/{roi_name}/best.ckpt")
    
    # Score good images — get baseline distribution
    good_scores = score_folder(model, f"data/rois/{roi_name}/good")
    
    # Score bad images — get anomaly distribution  
    bad_scores = score_folder(model, f"data/rois/{roi_name}/bad")
    
    # Plot histogram to visually verify separation
    plot_score_histogram(good_scores, bad_scores, roi_name)
    
    # Set threshold: 99th percentile of good scores
    # This means < 1% of good images will be false positives
    threshold = float(np.percentile(good_scores, 99))
    thresholds[roi_name] = threshold
    
    # Also compute AUROC for reporting
    auroc = compute_auroc(good_scores, bad_scores)
    print(f"{roi_name}: threshold={threshold:.4f}, AUROC={auroc:.3f}")

# Save thresholds for Person A to use in inference pipeline
with open("models/thresholds.json", "w") as f:
    json.dump(thresholds, f, indent=2)
print("Thresholds saved → models/thresholds.json")
```

**Expected `thresholds.json` output:**
```json
{
  "ROI_LOGO":       0.423,
  "ROI_INGREDIENT": 0.381,
  "ROI_GRAPHICS":   0.512,
  "ROI_BARCODE":    0.356
}
```

---

## Task B5: Evaluation Report (Day 3-4)

```python
# scripts/evaluate.py
# Run on held-out set (5 good + 5 bad per ROI that were NOT used in training/calibration)

metrics = {}
for roi_name in ROI_NAMES:
    threshold = thresholds[roi_name]
    tp, fp, fn, tn = 0, 0, 0, 0
    
    for img_path in held_out_good:
        score = model.predict(img_path)
        if score > threshold: fp += 1
        else: tn += 1
    
    for img_path in held_out_bad:
        score = model.predict(img_path)
        if score > threshold: tp += 1
        else: fn += 1  # missed defect!
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    metrics[roi_name] = {"AUROC": auroc, "F1": f1, "Recall": recall, "FPR": fp/(fp+tn)}
```

**Deliverables from Person B:**
- [ ] 4 trained EfficientAD model checkpoints in `models/`
- [ ] `models/thresholds.json` — threshold per ROI
- [ ] Score histogram plots (good vs bad separation)
- [ ] `scripts/evaluate.py` — evaluation script
- [ ] Evaluation report: AUROC, F1, Recall, FPR per ROI

---

# 🟢 Person C — Annotation + Validation

**Starts:** Day 1 (fully parallel)
**Total time:** ~2 days (critical path for threshold calibration)

Person C has the most time-critical role — Person B cannot calibrate thresholds until annotation is done.

---

## Task C1: Clarify 4 ROI Folder Contents (Day 1 — first 30 min)

Before annotating, understand exactly what each of the 4 folders contains:

```
Questions to answer:
1. What are the exact folder names?
2. Do the 4 folders have good/ and bad/ subfolders, or are they flat?
3. Are the bad crops named consistently with the original bad image numbers?
4. What are the dimensions of each crop (may differ per ROI)?
```

Share this info with Person B and Person A — they need it to match coordinates.

---

## Task C2: Annotate Bad ROI Crops with Bounding Boxes (Day 1-2)

> [!IMPORTANT]
> Annotate the **cropped ROI images** (not the full 1504×8000 images). Since the 4 folders are already cropped, annotate within those cropped images. This is much faster and coordinates will directly map to the model's input space.

**Tool: LabelImg (simpler, works well for small crops)**

```bash
pip install labelImg
labelImg
```

**Or use CVAT online (app.cvat.ai) for collaborative annotation.**

**Annotation classes:**
```
scratch        ← surface scratch
tear           ← label tear / cut
ink_blob       ← extra ink / black spot
smudge         ← smeared ink region
wrinkle        ← paper crinkle / fold
missing_print  ← region where ink is absent
unknown_defect ← if you can't classify it
```

**Process:**
```
For each ROI folder (4 folders):
  → Open bad/ subfolder in LabelImg
  → For each bad crop: draw ONE bounding box per visible defect
  → Label with defect class
  → Save as YOLO format (.txt files alongside images)
  → Move to next image
```

**Time estimate:** 42 bad images × 4 ROI crops × ~1 min each = ~3 hours total.

> [!TIP]
> In LabelImg, use keyboard shortcuts: W (create box), D (next image), A (previous image). This speeds up annotation significantly.

---

## Task C3: Create Held-Out Validation Set (Day 2)

After annotation, split your data to prevent threshold overfitting:

```
From 78 good ROI crops:
  → 68 for training (Person B uses these)
  → 10 held-out for evaluation (Person B's Task B5)

From 42 bad ROI crops (now annotated):
  → 32 for threshold calibration (Person B's Task B4)
  → 10 held-out for evaluation (Person B's Task B5)
```

Create a simple text file: `data/splits.json`
```json
{
  "train_good": ["001.png", "002.png", ...],
  "val_good":   ["069.png", "070.png", ...10 images...],
  "calib_bad":  ["001.png", "002.png", ...32 images...],
  "val_bad":    ["033.png", "034.png", ...10 images...]
}
```

---

## Task C4: Manual Visual Validation (Day 3-4)

After Person A has inference working:
1. Run inference on 5 good + 5 bad images
2. Look at the output images with bounding boxes
3. Answer these questions:
   - Do bounding boxes appear on bad images? (recall check)
   - Do bounding boxes appear on good images? (false positive check)
   - Is the bounding box roughly in the right location?
   - Are there any obvious wrong detections?
4. Report findings to Person B for threshold adjustment

**Deliverables from Person C:**
- [ ] YOLO annotation files for all 42 bad ROI crops (4 folders)
- [ ] `data/splits.json` — train/val/calib split
- [ ] Written description: what each of the 4 ROI folders contains
- [ ] Manual validation report (5 good + 5 bad visual check)

---

# Bounding Box Output Format

The final output of the system for each inspected image:

```json
{
  "image_path": "dataset/NSC/NSC BAD IMAGES/5.bmp",
  "overall_result": "FAIL",
  "timestamp": "2026-07-22T10:30:00",
  "roi_results": {
    "ROI_LOGO": {
      "anomaly_score": 0.72,
      "threshold": 0.42,
      "is_anomalous": true,
      "bounding_boxes": [
        {
          "x": 45, "y": 112, "w": 80, "h": 55,
          "score": 0.84,
          "area": 4400,
          "label": "anomaly"
        }
      ]
    },
    "ROI_INGREDIENT": {
      "anomaly_score": 0.21,
      "threshold": 0.38,
      "is_anomalous": false,
      "bounding_boxes": []
    },
    "ROI_GRAPHICS": {
      "anomaly_score": 0.18,
      "threshold": 0.51,
      "is_anomalous": false,
      "bounding_boxes": []
    },
    "ROI_BARCODE": {
      "anomaly_score": 0.11,
      "threshold": 0.36,
      "is_anomalous": false,
      "bounding_boxes": []
    }
  }
}
```

---

# Inference Script (runs on a full new image)

```python
# scripts/run_inference.py
import cv2, json, os
from src.pipeline import HimalayaInspector

inspector = HimalayaInspector(
    model_dir="models/",
    threshold_config="models/thresholds.json"
)

input_dir = "dataset/NSC/NSC BAD IMAGES"
output_dir = "outputs/inference_results"
os.makedirs(output_dir, exist_ok=True)

for fname in os.listdir(input_dir):
    if not fname.endswith('.bmp'):
        continue
    result = inspector.inspect(os.path.join(input_dir, fname))
    
    # Save annotated image
    out_img_path = os.path.join(output_dir, fname.replace('.bmp', '_result.png'))
    cv2.imwrite(out_img_path, result.annotated_image)
    
    # Save JSON result
    out_json_path = os.path.join(output_dir, fname.replace('.bmp', '_result.json'))
    with open(out_json_path, 'w') as f:
        json.dump(result.to_dict(), f, indent=2)
    
    verdict = "FAIL" if not result.overall_pass else "PASS"
    print(f"{fname}: {verdict}")
```

---

# What to Skip in This Sprint

| Item | Decision | Why |
|------|----------|-----|
| SegFormer pixel segmentation | ❌ Skip | Bounding boxes from heatmap are sufficient; SegFormer needs 200+ annotated crops |
| ORB alignment for training | ✅ Already done | ROI crops are pre-aligned; only needed for new inference images |
| Golden Master cosine similarity | ⚠️ Optional v2 | Per-ROI EfficientAD already does this more precisely |
| Left-Right SSIM | ❌ Skip | Not applicable to single-ROI crop approach |
| Barcode ZBar decode | ❌ Skip | Out of scope for defect detection demo |
| MLflow / experiment tracking | ❌ Skip | Not needed for demo |
| Full Streamlit dashboard | ✅ Basic version only | Just show image + bboxes + PASS/FAIL label |
| ONNX export | ❌ Skip | Only for edge deployment |

---

# Success Criteria

By end of sprint, you should be able to:

1. ✅ Feed a raw NSC BMP image into the system
2. ✅ Get a PASS or FAIL verdict
3. ✅ See bounding boxes drawn around detected anomaly regions
4. ✅ See which ROI (Logo / Ingredient / Graphics / Barcode) triggered the FAIL
5. ✅ Get a JSON file with bounding box coordinates and anomaly scores

**Target Metrics (first sprint):**
| Metric | Target |
|--------|--------|
| AUROC per ROI | > 0.85 |
| Recall (defect caught) | > 90% |
| False Positive Rate | < 10% |
| Inference time per image | < 200 ms |

---

# Open Questions (Need Answers Before Starting)

> [!IMPORTANT]
> Answer these before Day 1:

1. **What are the exact names of the 4 ROI folders?** (Logo, Ingredient, Graphics, Barcode — or different names?)
2. **Do the bad images already have bad/ subfolders in each ROI folder?** Or are all crops mixed?
3. **What are the pixel dimensions of each ROI crop?** (Needed by Person B for `image_size` param and Person A for coordinate mapping)
4. **Which person has access to the remote PC with the A500?** (Person B needs it for training; Person A may need it for inference pipeline testing)
5. **Do the ROI crop filenames match the original image numbers?** (e.g., does bad crop `5.png` correspond to `NSC BAD IMAGES/5.bmp`?)
