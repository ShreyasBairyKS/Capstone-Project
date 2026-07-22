"""
extract_bad_crops.py
─────────────────────────────────────────────────────────────────────────────
Uses template matching to locate each ROI in the 42 bad images, then
extracts and saves grayscale crops into the bad/ subfolders.

Run AFTER:
  1. save_templates.py has been run (patches exist in config/)
  2. validate_anchors.py confirms 100% pass rate on good images

Output structure:
  data/rois/ROI_LOGO/bad/*.png
  data/rois/ROI_INGREDIENT/bad/*.png
  data/rois/ROI_ADDRESS/bad/*.png

Usage:
    python extract_bad_crops.py
"""

import cv2
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "himalaya_label_detection"))
from src.preprocessing.anchor import TemplateAnchorFinder

CONFIG_DIR = Path("himalaya_label_detection/config")
BAD_DIR    = Path("dataset/NSC/NSC BAD IMAGES")
OUT_BASE   = Path("data/rois")

finder = TemplateAnchorFinder(CONFIG_DIR)

if not finder.templates:
    print("ERROR: No templates found. Run save_templates.py first.")
    sys.exit(1)

files = sorted(BAD_DIR.glob("*.bmp"))
print(f"Extracting crops from {len(files)} bad images...")
print(f"ROIs: {list(finder.templates.keys())}")
print()

skipped = 0; extracted = 0

for f in files:
    img = cv2.imread(str(f))   # load as BGR color
    if img is None:
        print(f"  ERROR: cannot read {f.name}")
        continue

    crops = finder.extract_crops(img)  # returns grayscale crops
    img_skipped = False

    for roi_name, crop in crops.items():
        if crop is None:
            print(f"  SKIP {f.name}/{roi_name}: anchor not found")
            img_skipped = True
            skipped += 1
            continue

        out_dir = OUT_BASE / roi_name / "bad"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / (f.stem + ".png")
        cv2.imwrite(str(out_path), crop)
        extracted += 1

    if not img_skipped:
        print(f"  OK   {f.name}: all 3 crops saved")

print()
print(f"Done. Extracted={extracted} crops | Skipped={skipped} (anchor not found)")
print(f"Output: {OUT_BASE}/ROI_*/bad/")
print()
print("Next step (Person C): annotate the bad crops using LabelImg")
print("  pip install labelImg && labelImg")
