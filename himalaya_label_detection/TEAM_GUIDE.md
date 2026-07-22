# Himalaya NSC Label Defect Detection
## Team Guide — 3 People, 5 Days

> Use AI (ChatGPT / Gemini) freely for any step.
> When stuck: paste the exact error message and ask AI to fix it.

---

## Project Summary

We inspect Himalaya Winter Defense Moisturizing Cream (50ml) tube labels for
defects (tears, smudges, missing print, ink blobs). Each image is an
unwrapped flat photo of the cylindrical tube — **1504 x 8000 px, RGB, BMP**.
The same label wraps around twice per image. The logo position shifts up to
**4620 px** across images, so we use **template matching** to find it.
We then crop 3 fixed-height ROI strips and run a dedicated **EfficientAD**
anomaly detection model on each strip.

---

## What We Have

| | Item | Location |
|---|---|---|
| 78 good full images | RGB BMP 1504x8000 | `dataset/NSC/NSC GOOD IMAGES/` |
| 42 bad full images | RGB BMP 1504x8000 | `dataset/NSC/NSC BAD IMAGES/` |
| 3 ROI folders | Grayscale crops, good+bad MIXED | Remote PC |
| Sample crops | 1.bmp (logo), 3.bmp (ingredient), 9.bmp (address) | Project root |
| A500 GPU | — | Remote PC |

**The 3 ROI folders already have good and bad images mixed together.**
Person C splits them into subfolders first, then annotates.

---

## 3 ROI Regions

```
Full image (8000px tall)
┌───────────────────────┐
│    [blank/dark area]  │
├───────────────────────┤ ← template match finds top of logo
│  ROI_LOGO   (~1398px) │  Himalaya logo + "Winter Defense" text
├───────────────────────┤
│ ROI_INGREDIENT(~1196) │  Jojoba/Wheat Germ/Almond Oil text block
├───────────────────────┤
│  ROI_ADDRESS (~1773px)│  Address + regulatory + MFG/EXP dates
├───────────────────────┤
│  [3 Way Care graphic] │  SKIPPED — no training crops for this
│  [repeats again below]│
└───────────────────────┘
```

**"3 Way Care" graphics panel is SKIPPED** (no crops available).

---

## Architecture

```
New RGB image (1504x8000)
        |
        v
  Template Matching  <-- finds each ROI independently, no training needed
  (src/preprocessing/anchor.py)
        |
        |-- crop ROI_LOGO       (grayscale) --> EfficientAD Model A --> score + heatmap --> bboxes
        |-- crop ROI_INGREDIENT (grayscale) --> EfficientAD Model B --> score + heatmap --> bboxes
        '-- crop ROI_ADDRESS    (grayscale) --> EfficientAD Model C --> score + heatmap --> bboxes
                                                      |
                                          PASS/FAIL + annotated RGB image + JSON
```

---

## 5-Day Timeline

```
Day 1  | A: Save template patches + validate ALL images pass
       | B: Set up GPU env on remote PC
       | C: Split ROI folders into good/ and bad/ subfolders

Day 2  | A: Update roi_coord_config.json with real heights
       | B: Verify folder structure, copy project to remote PC
       | C: Annotate bad crops with bounding boxes (start)

Day 3  | B: Train 3 EfficientAD models (~45 min on A500)
       | C: Finish annotation, share bad/ folders with B

Day 4  | B: Calibrate thresholds (once bad/ folders are ready)
       | A: Wire anchor.py into full inference pipeline

Day 5  | All: End-to-end test on full image set, tune thresholds
```

---

# PERSON A — Pipeline + Template Anchor

**PC:** Local (no GPU needed)

---

## A1. Setup

```powershell
cd "C:\Users\Ullas N\Desktop\Capstone-Project"
.venv\Scripts\activate
pip install opencv-python numpy
```

---

## A2. Save the 3 Template Patches (Day 1 — 10 min)

The sample crops (1.bmp, 3.bmp, 9.bmp) are in the project root.
Run once to extract reference patches for template matching:

```powershell
python himalaya_label_detection/scripts/save_templates.py
```

Creates:
- `himalaya_label_detection/config/patch_logo.png`
- `himalaya_label_detection/config/patch_ingredient.png`
- `himalaya_label_detection/config/patch_address.png`

---

## A3. Validate Template Matching on ALL 120 Images (Day 1 — 30 min)

```powershell
python validate_anchors.py
```

Expected output:
```
OK  1.bmp   LOGO=v 0.87 y=1478  INGREDIENT=v 0.71 y=2876  ADDRESS=v 0.68 y=4072
OK  2.bmp   LOGO=v 0.83 y= 420  ...
...
Good images: 78/78 fully found
Bad  images: 42/42 fully found
ALL GOOD IMAGES PASS - safe to proceed
```

**If any image FAILS (score < 0.50):**
- Lower `MATCH_THRESHOLD = 0.45` in `himalaya_label_detection/src/preprocessing/anchor.py`
- Or open `1.bmp` in Paint, choose a cleaner patch area, re-run `save_templates.py`
- Ask AI: *"My cv2.matchTemplate score is 0.40. How do I choose a better template patch?"*

---

## A4. Update ROI Heights in Config (Day 2)

Get the actual height of each pre-cropped folder's images from Person B/C:

```python
# Quick check — run on the remote PC or wherever crops are stored
import cv2
from pathlib import Path

for roi, folder in [
    ("ROI_LOGO",       Path("data/rois/ROI_LOGO/good")),
    ("ROI_INGREDIENT", Path("data/rois/ROI_INGREDIENT/good")),
    ("ROI_ADDRESS",    Path("data/rois/ROI_ADDRESS/good")),
]:
    for f in list(folder.glob("*"))[:1]:
        img = cv2.imread(str(f))
        if img is not None:
            print(f"{roi}: h={img.shape[0]}  w={img.shape[1]}")
```

Update `himalaya_label_detection/config/roi_coord_config.json` with the measured `h` values.

---

## A5. Wire Anchor into Inference Pipeline (Day 4)

`src/roi_pipeline.py` has the structure. Update the `inspect()` method
to call `TemplateAnchorFinder.extract_crops(rgb_image)` instead of fixed coords.

Test run:
```powershell
python himalaya_label_detection/scripts/run_roi_inference.py `
  --image "dataset\NSC\NSC BAD IMAGES\1.bmp" `
  --save-output outputs\test
```

**Person A deliverables:**
- [ ] `config/patch_logo.png`, `patch_ingredient.png`, `patch_address.png`
- [ ] `validate_anchors.py` 100% pass on all images
- [ ] `config/roi_coord_config.json` with real heights
- [ ] Working inference with annotated output image + JSON

---

# PERSON B — EfficientAD Training + Calibration

**PC:** Remote PC with A500 GPU

---

## B1. Clone and Setup (Day 1)

```bash
git clone https://github.com/ShreyasBairyKS/Capstone-Project.git
cd Capstone-Project
git checkout himalaya-label-detection

conda create -n himalaya python=3.10 -y
conda activate himalaya

# PyTorch with CUDA for A500
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Verify GPU
python -c "import torch; print('GPU:', torch.cuda.get_device_name(0))"

pip install anomalib==1.1.0
pip install opencv-python scikit-image pillow numpy tqdm matplotlib
```

---

## B2. Verify ROI Folder Structure (Day 2)

After Person C finishes sorting, your folders must look like:

```
data/rois/
  ROI_LOGO/
    good/     <- N grayscale crops from good images
    bad/      <- N grayscale crops from bad images
  ROI_INGREDIENT/
    good/
    bad/
  ROI_ADDRESS/
    good/
    bad/
```

Verify:
```python
import os
for roi in ["ROI_LOGO", "ROI_INGREDIENT", "ROI_ADDRESS"]:
    for split in ["good", "bad"]:
        p = f"data/rois/{roi}/{split}"
        n = len(os.listdir(p)) if os.path.exists(p) else 0
        print(f"{roi}/{split}: {n}")
```

Run verify script:
```bash
python himalaya_label_detection/scripts/organise_roi_folders.py \
  --rois-root data/rois --verify
```

---

## B3. Train All 3 EfficientAD Models (Day 3)

Training uses ONLY the `good/` images. Can start as soon as good/ folders are ready.

```bash
cd himalaya_label_detection

python scripts/train_all_rois.py \
  --rois-root data/rois \
  --models-root models/rois \
  --model efficientad \
  --epochs 100
```

~15 min per ROI = ~45 min total. If CUDA OOM: add `--batch-size 4`

---

## B4. Calibrate Thresholds (Day 4 - after Person C finishes bad/ folders)

```bash
python scripts/calibrate_roi_thresholds.py \
  --rois-root data/rois \
  --models-root models/rois \
  --target-recall 0.95 \
  --plot
```

Good result: green (good scores) and red (bad scores) histograms are clearly separated.
If they heavily overlap: retrain with `--epochs 200`.

Saves calibrated values to `config/roi_thresholds.json`.
Share this file + `models/rois/` with Person A.

**Person B deliverables:**
- [ ] 3 trained model folders in `models/rois/`
- [ ] `config/roi_thresholds.json` with calibrated values (not 0.5 defaults)
- [ ] Histogram plots showing score separation

---

# PERSON C — Folder Organisation + Annotation

**PC:** Remote PC (where the 3 ROI folders live)

---

## C1. Split Mixed ROI Folders into good/ and bad/ (Day 1 — 1 hour)

The 3 ROI folders currently have good and bad images MIXED together.
You need to separate them into subfolders.

**Step 1:** Create the subfolder structure:

```bash
python himalaya_label_detection/scripts/organise_roi_folders.py \
  --rois-root /path/to/your/roi/folders \
  --create-structure
```

This creates `good/` and `bad/` under each ROI folder.

**Step 2:** Open File Explorer (or a file manager) and go to each ROI folder.
Look at each image carefully:

| What you see | Move to |
|-------------|---------|
| Clean, undamaged label print | `good/` |
| Tear, smudge, missing print, ink blob, wrinkle | `bad/` |
| Not sure | `bad/` (safer to over-flag) |

Do this for all 3 ROI folders:
- `/path/to/ROI_LOGO/` → sort into `ROI_LOGO/good/` and `ROI_LOGO/bad/`
- `/path/to/ROI_INGREDIENT/` → sort into `ROI_INGREDIENT/good/` and `ROI_INGREDIENT/bad/`
- `/path/to/ROI_ADDRESS/` → sort into `ROI_ADDRESS/good/` and `ROI_ADDRESS/bad/`

**Step 3:** Verify counts:

```bash
python himalaya_label_detection/scripts/organise_roi_folders.py \
  --rois-root /path/to/your/roi/folders \
  --verify
```

Expected:
```
ROI_LOGO/:
    good/ : 78 images
    bad/  : 42 images

ROI_INGREDIENT/:
    good/ : 78 images
    bad/  : 42 images

ROI_ADDRESS/:
    good/ : 78 images
    bad/  : 42 images

All folders correctly organised. Ready for training.
```

---

## C2. Annotate Bad Crops with Bounding Boxes (Day 2-3 — ~2 hours)

Install LabelImg:
```bash
pip install labelImg
labelImg
```

> If it crashes: use https://app.cvat.ai (free, no install needed)

In LabelImg:
1. **Open Dir** -> point to `ROI_LOGO/bad/`
2. Format: **YOLO** (select from left panel)
3. **Change Save Dir** -> `data/annotations/ROI_LOGO/`
4. Draw boxes around every defect:
   - `W` = draw box
   - `D` = next image
   - `A` = previous image
   - `Ctrl+S` = save

Label classes to use:
- `scratch`
- `tear`
- `ink_blob`
- `smudge`
- `wrinkle`
- `missing_print`
- `unknown_defect`

Repeat for `ROI_INGREDIENT/bad/` and `ROI_ADDRESS/bad/`.

---

## C3. Visual Validation (Day 5)

After Person A runs full inference, check 10-15 output images:

For bad images: are the red bounding boxes where defects actually are?
For good images: all show PASS with no boxes?

Write notes:
```
1.bmp:  FAIL at ROI_LOGO - box at y=340, smudge visible there ✓
5.bmp:  FAIL at ROI_ADDRESS - box on barcode tear ✓
10.bmp: FAIL at ROI_LOGO - FALSE POSITIVE, no visible defect ✗ (tell B to raise threshold)
```

**Person C deliverables:**
- [ ] All 3 ROI folders sorted into good/ and bad/ subfolders
- [ ] `data/annotations/ROI_*/` - YOLO .txt annotation files
- [ ] Validation notes from Day 5

---

## Common Problems and Fixes

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError: anomalib` | `pip install anomalib==1.1.0` |
| Template score < 0.50 | Lower `MATCH_THRESHOLD = 0.45` in `anchor.py` |
| CUDA not detected | Ask AI: *"torch.cuda.is_available() False, NVIDIA A500, Windows"* |
| LabelImg crashes | Use https://app.cvat.ai instead |
| No bboxes on bad images | Person B: lower values in `config/roi_thresholds.json` |
| Too many false bboxes on good images | Person B: raise values in `roi_thresholds.json` |
| Git push rejected | Run `git pull origin himalaya-label-detection` first |

---

## Key Commands

```powershell
# Person A - Day 1
python himalaya_label_detection/scripts/save_templates.py
python validate_anchors.py

# Person C - Day 1 (on remote PC)
python himalaya_label_detection/scripts/organise_roi_folders.py --rois-root /path/to/rois --create-structure
# Sort images manually in File Explorer
python himalaya_label_detection/scripts/organise_roi_folders.py --rois-root /path/to/rois --verify

# Person C - Day 2-3
labelImg   # annotate bad crops

# Person B - Day 3 (on remote PC with A500)
python himalaya_label_detection/scripts/train_all_rois.py --rois-root data/rois --models-root models/rois

# Person B - Day 4
python himalaya_label_detection/scripts/calibrate_roi_thresholds.py --plot

# Person A - Day 4-5
python himalaya_label_detection/scripts/run_roi_inference.py --folder "dataset/NSC/NSC BAD IMAGES" --save-output outputs/
```

---

## Git

```powershell
git pull origin himalaya-label-detection   # get latest
git add .
git commit -m "brief description of change"
git push origin himalaya-label-detection
```

Repo: https://github.com/ShreyasBairyKS/Capstone-Project
Branch: himalaya-label-detection
