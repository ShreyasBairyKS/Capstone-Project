"""
augment_dataset.py
──────────────────
Optional pre-training augmentation to expand the small ROI_2 good-image
set (~77 crops) before feeding into StyleGAN2-ADA.

Applies a variety of augmentations to each image and writes the results
alongside the originals in the same images/ folder.

This is especially useful when you have < 100 real images, as it gives
StyleGAN2-ADA a larger and more varied starting distribution.

Augmentations applied (all label-safe — no geometric warps that would
corrupt text legibility):
  - Horizontal flip
  - Slight brightness / contrast jitter
  - Gaussian blur (mild)
  - JPEG compression artifacts
  - Slight hue/saturation shift

Usage (from project root on remote PC):
    python himalaya_label_detection/Image_generator/ROI_2/augment_dataset.py

Run this BEFORE prepare_dataset.py (or after — it reads from SOURCE_DIR
and writes back to SOURCE_DIR with aug_ prefix).
"""

import random
from pathlib import Path

import cv2
import numpy as np

# ── Config ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[3]
SOURCE_DIR   = PROJECT_ROOT / "data" / "rois" / "ROI_2" / "good"
AUGMENTS_PER_IMAGE = 5      # number of augmented variants per real image
SEED = 42
# ─────────────────────────────────────────────────────────────────────────────


def brightness_contrast(img: np.ndarray) -> np.ndarray:
    alpha = random.uniform(0.80, 1.20)   # contrast
    beta  = random.randint(-20, 20)      # brightness
    return np.clip(alpha * img.astype(np.float32) + beta, 0, 255).astype(np.uint8)


def gaussian_blur(img: np.ndarray) -> np.ndarray:
    k = random.choice([0, 0, 3])         # mostly skip; occasionally blur
    if k == 0:
        return img
    return cv2.GaussianBlur(img, (k, k), 0)


def jpeg_artifact(img: np.ndarray) -> np.ndarray:
    quality = random.randint(70, 95)
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def hue_saturation_shift(img: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int32)
    hsv[:, :, 0] = np.clip(hsv[:, :, 0] + random.randint(-8, 8),  0, 179)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] + random.randint(-20, 20), 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def horizontal_flip(img: np.ndarray) -> np.ndarray:
    # NOTE: For label text this makes text unreadable — use only if that is
    # acceptable for your training purpose. Set ALLOW_HFLIP = False to skip.
    return cv2.flip(img, 1)


ALLOW_HFLIP = False   # ← set True only if mirrored text is acceptable


def augment(img: np.ndarray) -> np.ndarray:
    """Apply a random chain of augmentations."""
    ops = [brightness_contrast, gaussian_blur, jpeg_artifact, hue_saturation_shift]
    if ALLOW_HFLIP:
        ops.append(horizontal_flip)
    random.shuffle(ops)
    for op in ops[:random.randint(2, 4)]:
        img = op(img)
    return img


def main():
    random.seed(SEED)
    np.random.seed(SEED)

    if not SOURCE_DIR.exists():
        raise FileNotFoundError(f"Source directory not found: {SOURCE_DIR}")

    originals = sorted(SOURCE_DIR.glob("*.png")) + \
                sorted(SOURCE_DIR.glob("*.PNG")) + \
                sorted(SOURCE_DIR.glob("*.jpg")) + \
                sorted(SOURCE_DIR.glob("*.bmp"))

    # Skip already-augmented files
    originals = [p for p in originals if not p.stem.startswith("aug_")]

    print(f"Found {len(originals)} original images.")
    print(f"Generating {AUGMENTS_PER_IMAGE} augmented variants each → "
          f"{len(originals) * AUGMENTS_PER_IMAGE} new images.\n")

    created = 0
    for fp in originals:
        img = cv2.imread(str(fp))
        if img is None:
            continue
        for j in range(AUGMENTS_PER_IMAGE):
            aug = augment(img.copy())
            out_name = f"aug_{fp.stem}_{j:02d}.png"
            out_path = SOURCE_DIR / out_name
            cv2.imwrite(str(out_path), aug)
            created += 1

    print(f"✅  Done. Created {created} augmented images in:\n   {SOURCE_DIR}")
    print(f"\nTotal dataset size now: {len(originals) + created} images")
    print("Next step → run: prepare_dataset.py")


if __name__ == "__main__":
    main()
