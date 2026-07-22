# Himalaya NSC Label Defect Detection
## Complete Step-by-Step Guide for All 3 Team Members

> **You can use AI (ChatGPT / Gemini) to help with any step.**
> If something fails, copy-paste the error message into AI and ask for help.
> This guide assumes basic Python knowledge.

---

## Project Overview (Read This First)

**What we are building:** A system that looks at photos of Himalaya cream tube labels and automatically detects defects (tears, smudges, missing print, ink blobs, etc.)

**Dataset facts:**
- Full NSC images: **RGB** color, 1504x8000 px BMP (unwrapped cylindrical label)
- `dataset/NSC/NSC GOOD IMAGES/` - 78 good images
- `dataset/NSC/NSC BAD IMAGES/` - 42 bad images
- 3 ROI folders on remote PC with **grayscale** cropped regions:
  - `ROI_LOGO/good/` - logo area crops from good images
  - `ROI_INGREDIENT/good/` - ingredient strip crops from good images
  - `ROI_BARCODE/good/` - barcode/batch area crops from good images
- Each folder needs a `bad/` subfolder (Person C creates this)

**4-Day Plan:**
```
Day 1-2: Person B trains models | Person C annotates + crops | Person A builds pipeline
Day 3:   Person C hands bad crops to B -> B calibrates thresholds -> A integrates
Day 4:   All 3 test together, fix issues
```

---

# PERSON A - Inference Pipeline + Bounding Box Output

**Goal:** Build the code that takes a new label photo, runs trained models, draws red boxes around defects.

**Start Day 1** - No need to wait for training. Use dummy outputs first.

---

## A-Step 1: Open the Project in VS Code

1. Open VS Code
2. File -> Open Folder -> `C:\Users\Ullas N\Desktop\Capstone-Project`
3. Open a terminal in VS Code: View -> Terminal
4. Activate the virtual environment:
   ```powershell
   .venv\Scripts\activate
   ```
5. Install required packages:
   ```powershell
   pip install opencv-python numpy pillow
   ```

---

## A-Step 2: Understand What Was Already Built

Go to `himalaya_label_detection/` and read these files:

- `src/roi_pipeline.py` - The main inspection logic
- `src/postprocessing/bbox.py` - Converts heatmaps to bounding boxes
- `scripts/run_roi_inference.py` - The script you run from terminal

Ask AI to explain any part you don't understand:
*"Explain this Python code to me: [paste the code]"*

---

## A-Step 3: Test the Pipeline Loads Correctly

```powershell
cd himalaya_label_detection
python scripts/run_roi_inference.py --help
```

You should see usage instructions. If you get errors, ask AI:
*"I got this Python error: [paste error]. How do I fix it?"*

---

## A-Step 4: Fill in ROI Coordinates (DO THIS DAY 1)

Open `config/roi_coord_config.json`. It has PLACEHOLDER values like `"y": 0`. 
You must measure the REAL pixel coordinates.

**How to measure using GIMP (free image editor):**

1. Download GIMP from gimp.org
2. Open: File -> Open -> `dataset/NSC/NSC GOOD IMAGES/1.bmp`
3. Go to: Windows -> Dockable Dialogs -> Pointer Information
4. Hover your mouse over the image - you will see X and Y coordinates
5. Find where each region starts and ends

What to measure:
- `ROI_LOGO`: Top-left corner and size of the Himalaya logo + "Winter Defense" text
- `ROI_INGREDIENT`: "Jojoba Oil" ingredient strip area
- `ROI_BARCODE`: The barcode lines + MFG/EXP date area

Edit the JSON with your measurements (replace 0 values with real numbers).

**Verify your coordinates with this script:**

```python
# Save as verify_coords.py in project root
import cv2, json

img = cv2.imread(r"dataset\NSC\NSC GOOD IMAGES\1.bmp")
with open(r"himalaya_label_detection\config\roi_coord_config.json") as f:
    coords = {k: v for k, v in json.load(f).items() if not k.startswith("_")}

colors = {"ROI_LOGO": (0,0,255), "ROI_INGREDIENT": (0,165,255), "ROI_BARCODE": (0,200,0)}
for name, c in coords.items():
    if "x" in c:
        cv2.rectangle(img, (c["x"], c["y"]), (c["x"]+c["w"], c["y"]+c["h"]), colors.get(name,(255,255,0)), 10)
        cv2.putText(img, name, (c["x"]+10, c["y"]+80), cv2.FONT_HERSHEY_SIMPLEX, 3, colors.get(name,(255,255,0)), 5)

scale = 0.12
small = cv2.resize(img, (int(img.shape[1]*scale), int(img.shape[0]*scale)))
cv2.imwrite("coord_verify.png", small)
print("Saved coord_verify.png - open it to check boxes are in the right place")
```

Run: `python verify_coords.py` then open `coord_verify.png`

---

## A-Step 5: Plug in Real Models (Day 3)

When Person B finishes:
1. Copy `models/rois/` folder from remote PC to `himalaya_label_detection/models/rois/`
2. Copy `roi_thresholds.json` to `himalaya_label_detection/config/roi_thresholds.json`
3. Run on bad images:

```powershell
cd himalaya_label_detection
python scripts/run_roi_inference.py `
  --folder "..\dataset\NSC\NSC BAD IMAGES" `
  --save-output "..\outputs\bad_results" `
  --json-out "..\outputs\bad_results.json"
```

4. Open images in `outputs/bad_results/` - do you see red boxes on defects?

**Person A Deliverables:**
- [ ] `config/roi_coord_config.json` with real pixel coordinates
- [ ] `coord_verify.png` showing boxes are in right positions
- [ ] Output images with bounding boxes drawn on bad images
- [ ] JSON results file showing scores and bbox coordinates

---

# PERSON B - Model Training + Threshold Calibration

**Goal:** Train 3 EfficientAD models on the remote A500 GPU PC.

**Work on the REMOTE PC with A500 GPU.**

---

## B-Step 1: Copy Code to Remote PC

```bash
# On remote PC terminal:
git clone https://github.com/ShreyasBairyKS/Capstone-Project.git
cd Capstone-Project/himalaya_label_detection
```

Or copy the folder via USB.

---

## B-Step 2: Set Up Python Environment on Remote PC

```bash
# Create conda environment
conda create -n himalaya python=3.10 -y
conda activate himalaya

# Install PyTorch with GPU support (for A500 with CUDA 11.8)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Verify GPU works
python -c "import torch; print('GPU:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# Should print: GPU: True  NVIDIA A500 ...

# Install anomalib
pip install anomalib==1.1.0

# Install other packages
pip install opencv-python scikit-image pillow numpy tqdm matplotlib
```

If GPU is not detected, ask AI:
*"torch.cuda.is_available() is False on a PC with NVIDIA A500 GPU. What should I check?"*

---

## B-Step 3: Verify ROI Folder Structure

Your 3 folders should look like this BEFORE training:
```
data/rois/
  ROI_LOGO/
    good/      <- grayscale crops from 78 good images (ALREADY EXISTS)
    bad/       <- grayscale crops from 42 bad images (ADD THIS - from Person C)
  ROI_INGREDIENT/
    good/
    bad/
  ROI_BARCODE/
    good/
    bad/
```

Run this check:
```python
import os
rois_root = "data/rois"  # change to your actual path
for name in sorted(os.listdir(rois_root)):
    d = os.path.join(rois_root, name)
    g = len(os.listdir(os.path.join(d,"good"))) if os.path.exists(os.path.join(d,"good")) else 0
    b = len(os.listdir(os.path.join(d,"bad")))  if os.path.exists(os.path.join(d,"bad"))  else 0
    print(f"{name}: good={g} bad={b}")
```

Expected: `good=78, bad=42` for each of the 3 folders.

**NOTE: Do NOT start calibration without bad/ folders. You can start TRAINING without them.**

---

## B-Step 4: Train EfficientAD Models (Start Day 1)

```bash
python scripts/train_all_rois.py \
  --rois-root data/rois \
  --models-root models/rois \
  --model efficientad \
  --epochs 100
```

What you should see on screen:
```
===========================================================
  HIMALAYA NSC - PER-ROI ANOMALY MODEL TRAINING
===========================================================

  Training: ROI_LOGO
  ─────────────────
  [ROI_LOGO] Good crops: 78  |  Bad crops: 0
  [ROI_LOGO] Model: EfficientAD-SMALL  |  Epochs: 100
  [ROI_LOGO] Starting training...
  Epoch 1/100: ...
  ...
  [ROI_LOGO] Training complete
```

**Estimated time:** ~15 min per ROI x 3 = ~45 min total on A500 GPU.

If training crashes with "CUDA out of memory":
```bash
python scripts/train_all_rois.py --rois-root data/rois --models-root models/rois --batch-size 4
```

---

## B-Step 5: Calibrate Thresholds (Day 3 - After Person C adds bad crops)

```bash
python scripts/calibrate_roi_thresholds.py \
  --rois-root data/rois \
  --models-root models/rois \
  --target-recall 0.95 \
  --plot
```

Output shows:
```
ROI_LOGO:       threshold=0.4230  Recall=0.952  FPR=0.051
ROI_INGREDIENT: threshold=0.3810  Recall=0.961  FPR=0.037
ROI_BARCODE:    threshold=0.3560  Recall=0.976  FPR=0.029
```

**Reading the histogram plots:**
- Green histogram = good image scores (should be LOW)
- Red histogram = bad image scores (should be HIGH)
- Black vertical line = threshold

If the two histograms heavily OVERLAP -> model did not learn well -> retrain with more epochs (200)
If they are SEPARATED cleanly -> excellent, the model works

**Person B Deliverables:**
- [ ] 3 trained model checkpoint folders in `models/rois/`
- [ ] `config/roi_thresholds.json` with calibrated values (not 0.5 defaults)
- [ ] Histogram plot images showing score separation
- [ ] Transfer `models/rois/` to Person A and push to GitHub

---

# PERSON C - Annotation + Bad Crops + Validation

**Goal:** Annotate 42 bad images, extract bad ROI crops, validate final results.

---

## C-Step 1: Install LabelImg for Annotation

```powershell
pip install labelImg
labelImg
```

If this fails, use the free online tool: https://app.cvat.ai (no install needed)

---

## C-Step 2: Annotate 42 Bad Images

In LabelImg:
1. Open Dir -> `dataset/NSC/NSC BAD IMAGES`
2. Change Save Dir -> `data/annotations/labels` (create this folder first)
3. Format: select **YOLO** on the left panel
4. Add these label names: `scratch`, `tear`, `ink_blob`, `smudge`, `wrinkle`, `missing_print`, `unknown_defect`

**Keyboard shortcuts (makes it fast):**
- `W` = draw bounding box
- `D` = next image
- `A` = previous image
- `Del` = delete selected box
- `Ctrl+S` = save

**For each image:**
1. Look at the label carefully
2. Draw a box around each visible defect
3. Pick the class that best describes it (use `unknown_defect` if unsure)
4. Press `Ctrl+S` to save, then `D` for next image

Time estimate: ~2-3 minutes per image x 42 = about 2 hours total.

> **Tip:** Ask AI to show you examples: *"What does a label smudge defect look like vs a tear?"*

---

## C-Step 3: Extract Bad ROI Crops and Transfer to Remote PC

After Person A fills in `config/roi_coord_config.json`, run this crop script:

```python
# Save as: extract_bad_crops.py in the project root
import cv2, json, os

# Load ROI coordinates from Person A
with open(r"himalaya_label_detection\config\roi_coord_config.json") as f:
    coords = {k: v for k, v in json.load(f).items() if not k.startswith("_")}

bad_dir = r"dataset\NSC\NSC BAD IMAGES"
out_base = r"data\rois"

for roi_name, c in coords.items():
    out_dir = os.path.join(out_base, roi_name, "bad")
    os.makedirs(out_dir, exist_ok=True)

for fname in sorted(os.listdir(bad_dir)):
    if not fname.endswith(".bmp"):
        continue
    img = cv2.imread(os.path.join(bad_dir, fname))  # load as RGB
    for roi_name, c in coords.items():
        crop = img[c["y"]:c["y"]+c["h"], c["x"]:c["x"]+c["w"]]
        gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)  # save as grayscale
        out_path = os.path.join(out_base, roi_name, "bad", fname.replace(".bmp", ".png"))
        cv2.imwrite(out_path, gray_crop)
    print(f"Cropped: {fname}")

print("Done! Transfer data/rois/*/bad/ folders to the remote PC")
```

Run:
```powershell
python extract_bad_crops.py
```

Then copy the `data/rois/` folder to the remote PC (via USB drive or file sharing).

---

## C-Step 4: Visual Validation (Day 4)

After Person A runs the inference:

1. Open 5-10 images from `outputs/bad_results/`
2. Check: are the red bounding boxes in the right place?
3. Open 5 images from `outputs/good_results/` (you may need to run inference on good images too)
4. Check: do good images show PASS with no boxes?

Write a short validation note:
```
Image 1.bmp: FAIL - box at logo area - correct, there is a smudge there ✓
Image 5.bmp: FAIL - box at barcode - correct, barcode is torn ✓
Image 10.bmp: FAIL - box at ingredient bar - FALSE POSITIVE, no visible defect ✗
Good image 1.bmp: PASS - no boxes - correct ✓
```

Share this with Person B to help adjust thresholds.

**Person C Deliverables:**
- [ ] `data/annotations/labels/` - YOLO annotation .txt files for all 42 bad images
- [ ] `data/rois/ROI_*/bad/` - grayscale crops extracted and transferred to remote PC
- [ ] ROI coordinates measured and shared with Person A
- [ ] Validation notes (which images were correct/wrong)

---

# Common Problems and Solutions

| Problem | What to do |
|---------|-----------|
| `ModuleNotFoundError: anomalib` | `pip install anomalib==1.1.0` |
| `torch.cuda.is_available()` is False | Check NVIDIA drivers. Ask AI with your GPU name |
| LabelImg crashes on startup | Use https://app.cvat.ai instead |
| No boxes appear on bad images | Person B: lower threshold values in roi_thresholds.json |
| Too many false boxes on good images | Person B: raise threshold values |
| `cv2.error: (-215)` when opening BMP | Try `cv2.imread(path, cv2.IMREAD_COLOR)` |
| Git push rejected | Run `git pull origin himalaya-label-detection` first |

---

# How to Use AI for Help

When you are stuck, give AI this context:

*"I am working on a Himalaya label defect detection project in Python. We use anomalib EfficientAD to train anomaly detection models on 3 ROI crops (grayscale images of logo, ingredient bar, and barcode regions). The full images are RGB 1504x8000 BMP files. [describe your specific problem]. Error: [paste the exact error message]"*

Good AI tools: ChatGPT (chat.openai.com), Gemini (gemini.google.com), Claude

---

# Git Quick Reference

```bash
# Get latest changes from team
git pull origin himalaya-label-detection

# See what files you changed
git status

# Save your work
git add .
git commit -m "Add: brief description of what you did"
git push origin himalaya-label-detection
```

GitHub: https://github.com/ShreyasBairyKS/Capstone-Project
Branch: himalaya-label-detection
