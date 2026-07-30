"""
create_grayscale_dataset.py
─────────────────────────────────────────────────────────────────────────────
Converts the colored crops in data/rois/ into a grayscale dataset in
data/gray_scale_rois/.

The images are converted to grayscale and then replicated to 3 channels (BGR)
so that EfficientAD (which expects 3-channel input) can train on them natively
without modifying its architecture.

Usage:
    python himalaya_label_detection/scripts/create_grayscale_dataset.py
"""

import cv2
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
IN_DIR = PROJECT_ROOT / "data" / "rois"
OUT_DIR = PROJECT_ROOT / "data" / "gray_scale_rois"

SUPPORTED = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

def main():
    if not IN_DIR.exists():
        sys.exit(f"❌ Input directory not found: {IN_DIR}")

    rois = [d for d in IN_DIR.iterdir() if d.is_dir() and d.name.startswith("ROI_")]
    if not rois:
        sys.exit(f"❌ No ROI folders found in {IN_DIR}")

    total_converted = 0

    for roi in rois:
        print(f"Processing {roi.name}...")
        for split in ["good", "bad", "ground_truth/bad"]:
            in_split = roi / split
            if not in_split.exists():
                continue

            out_split = OUT_DIR / roi.name / split
            out_split.mkdir(parents=True, exist_ok=True)

            files = [f for f in in_split.iterdir() if f.suffix.lower() in SUPPORTED]
            for f in files:
                # If it's a ground truth mask, just copy it
                if split == "ground_truth/bad":
                    img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
                    cv2.imwrite(str(out_split / f.name), img)
                    continue

                # Load color image
                img_bgr = cv2.imread(str(f))
                if img_bgr is None:
                    print(f"  [WARN] Failed to load {f}")
                    continue

                # Convert to grayscale
                gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
                # Replicate to 3 channels so EfficientAD accepts it natively
                gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

                cv2.imwrite(str(out_split / f.name), gray_3ch)
                total_converted += 1
                
        print(f"  [OK] {roi.name} done.")

    print(f"\n✅ Converted {total_converted} images to grayscale.")
    print(f"Dataset saved to: {OUT_DIR}")
    print("\nNext step: Train on the grayscale dataset")
    print(f"  python himalaya_label_detection/scripts/train_classifiers.py --rois data/gray_scale_rois --out models/gray_scale_rois")

if __name__ == "__main__":
    main()
