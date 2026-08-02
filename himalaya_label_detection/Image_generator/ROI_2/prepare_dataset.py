"""
prepare_dataset.py
──────────────────
Converts ROI_2 good PNG crops into a StyleGAN2-ADA training dataset.

Steps performed:
  1. Reads all PNGs from data/rois/ROI_2/good/
  2. Pads / resizes each image to TARGET_RES x TARGET_RES (square) while
     preserving aspect ratio (letterbox with black padding)
  3. Writes images into a flat directory  →  out_dir/
  4. Creates a dataset.json  (StyleGAN2-ADA format: {"labels": []})
  5. Zips everything into  roi2_good_256.zip  ready for training

Usage (from project root on remote PC):
    python himalaya_label_detection/Image_generator/ROI_2/prepare_dataset.py

Outputs:
    himalaya_label_detection/Image_generator/ROI_2/dataset/roi2_good_256.zip
"""

import json
import zipfile
from pathlib import Path

import cv2
import numpy as np

# ── Config ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[3]          # project root
SOURCE_DIR   = PROJECT_ROOT / "data" / "rois" / "ROI_2" / "good"
OUT_DIR      = Path(__file__).parent / "dataset" / "images"
ZIP_OUT      = Path(__file__).parent / "dataset" / "roi2_good_256.zip"
TARGET_RES   = 256          # StyleGAN2 requires power-of-2 square images
# ─────────────────────────────────────────────────────────────────────────────


def letterbox(img: np.ndarray, size: int) -> np.ndarray:
    """Resize image to `size x size` with black letterbox padding."""
    h, w = img.shape[:2]
    scale = size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    pad_top  = (size - new_h) // 2
    pad_left = (size - new_w) // 2
    canvas[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized
    return canvas


def main():
    if not SOURCE_DIR.exists():
        raise FileNotFoundError(
            f"Source directory not found: {SOURCE_DIR}\n"
            f"Expected PNG crops at: data/rois/ROI_2/good/"
        )

    png_files = sorted(SOURCE_DIR.glob("*.png"))
    if not png_files:
        # Also accept other image formats
        png_files = sorted(SOURCE_DIR.glob("*.PNG")) + \
                    sorted(SOURCE_DIR.glob("*.jpg")) + \
                    sorted(SOURCE_DIR.glob("*.bmp"))

    print(f"Found {len(png_files)} images in {SOURCE_DIR}")
    if len(png_files) < 10:
        print("⚠️  WARNING: StyleGAN2-ADA works best with ≥50 images. "
              "Consider augmenting first.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    processed = []
    for i, fp in enumerate(png_files):
        img = cv2.imread(str(fp))
        if img is None:
            print(f"  [SKIP] Cannot read: {fp.name}")
            continue

        # Ensure 3-channel BGR
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        squared = letterbox(img, TARGET_RES)

        out_name = f"roi2_{i:04d}.png"
        out_path = OUT_DIR / out_name
        cv2.imwrite(str(out_path), squared)   # write BGR directly — no double conversion
        processed.append(out_name)
        print(f"  [{i+1:3d}/{len(png_files)}] {fp.name} → {out_name}")

    # Write dataset.json  (StyleGAN2-ADA expects this; empty labels = unconditional)
    dataset_meta = {"labels": []}
    json_path = OUT_DIR / "dataset.json"
    json_path.write_text(json.dumps(dataset_meta, indent=2))

    # Zip everything
    print(f"\nCreating ZIP: {ZIP_OUT}")
    ZIP_OUT.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(str(ZIP_OUT), "w", compression=zipfile.ZIP_STORED) as zf:
        for fname in processed:
            zf.write(str(OUT_DIR / fname), arcname=fname)
        zf.write(str(json_path), arcname="dataset.json")

    print(f"\n✅  Dataset ready: {ZIP_OUT}")
    print(f"   Images: {len(processed)}  |  Resolution: {TARGET_RES}x{TARGET_RES}")
    print(f"\nNext step → run: train_stylegan.py")


if __name__ == "__main__":
    main()
