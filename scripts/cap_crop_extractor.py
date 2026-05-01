"""
scripts/cap_crop_extractor.py
==============================
Extract cap crops from detection results for classification training.

Two modes:
  1. FROM YOLO LABELS: Read existing YOLO label .txt files and crop cap regions
  2. FROM MODEL: Run a trained detector and crop detected caps

Preserves aspect ratio using letterbox padding (no distortion).
Removes near-duplicate crops using perceptual hashing.

Usage:
    # From existing YOLO labels (your Roboflow dataset):
    python scripts/cap_crop_extractor.py \
        --mode labels \
        --images dataset/Beverages/bottleDefect.v1-first.yolov11-cap/train/images \
        --labels dataset/Beverages/bottleDefect.v1-first.yolov11-cap/train/labels \
        --output data/caps/unsorted \
        --cap-class-ids 0,1 \
        --size 224

    # From a trained detector model:
    python scripts/cap_crop_extractor.py \
        --mode model \
        --images path/to/images \
        --weights runs/detect/bottle_cap_det_v2/weights/best.pt \
        --output data/caps/unsorted \
        --cap-class-name cap \
        --size 224
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def letterbox_crop(crop: np.ndarray, size: int) -> np.ndarray:
    """Resize crop preserving aspect ratio with gray padding."""
    h, w = crop.shape[:2]
    ratio = min(size / h, size / w)
    new_h, new_w = int(round(h * ratio)), int(round(w * ratio))
    resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_top = (size - new_h) // 2
    pad_left = (size - new_w) // 2
    padded = cv2.copyMakeBorder(
        resized,
        pad_top, size - new_h - pad_top,
        pad_left, size - new_w - pad_left,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )
    return padded


def perceptual_hash(img: np.ndarray, hash_size: int = 8) -> str:
    """Simple average hash for deduplication."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    resized = cv2.resize(gray, (hash_size, hash_size), interpolation=cv2.INTER_AREA)
    mean = resized.mean()
    bits = (resized > mean).flatten()
    hash_int = 0
    for bit in bits:
        hash_int = (hash_int << 1) | int(bit)
    return f"{hash_int:016x}"


def extract_from_labels(
    images_dir: Path,
    labels_dir: Path,
    output_dir: Path,
    cap_class_ids: set[int],
    size: int,
    context_pad: float,
) -> int:
    """Extract cap crops using YOLO label files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    seen_hashes: set[str] = set()
    count = 0
    dupes = 0

    image_files = sorted(
        p for p in images_dir.glob("*.*")
        if p.suffix.lower() in IMAGE_EXTENSIONS
    )

    for img_path in image_files:
        lbl_path = labels_dir / f"{img_path.stem}.txt"
        if not lbl_path.exists():
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        for line in lbl_path.read_text(encoding="utf-8").strip().splitlines():
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            cls_id = int(float(parts[0]))
            if cls_id not in cap_class_ids:
                continue

            cx, cy, bw, bh = map(float, parts[1:5])

            # Add context padding
            pad_w = bw * context_pad
            pad_h = bh * context_pad
            x1 = max(0, int((cx - bw / 2 - pad_w) * w))
            y1 = max(0, int((cy - bh / 2 - pad_h) * h))
            x2 = min(w, int((cx + bw / 2 + pad_w) * w))
            y2 = min(h, int((cy + bh / 2 + pad_h) * h))

            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            # Letterbox resize
            crop_resized = letterbox_crop(crop, size)

            # Deduplication
            phash = perceptual_hash(crop_resized)
            if phash in seen_hashes:
                dupes += 1
                continue
            seen_hashes.add(phash)

            out_name = f"{img_path.stem}_cls{cls_id}_crop{count}.jpg"
            cv2.imwrite(
                str(output_dir / out_name),
                crop_resized,
                [cv2.IMWRITE_JPEG_QUALITY, 95],
            )
            count += 1

    return count


def extract_from_model(
    images_dir: Path,
    weights_path: Path,
    output_dir: Path,
    cap_class_name: str,
    size: int,
    conf: float,
    context_pad: float,
) -> int:
    """Extract cap crops using a trained YOLOv8 detector."""
    from ultralytics import YOLO

    output_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(weights_path))
    seen_hashes: set[str] = set()
    count = 0

    image_files = sorted(
        p for p in images_dir.glob("*.*")
        if p.suffix.lower() in IMAGE_EXTENSIONS
    )

    for img_path in image_files:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        results = model.predict(str(img_path), conf=conf, verbose=False)
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            cls_name = results[0].names.get(cls_id, str(cls_id))

            if cls_name != cap_class_name:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            # Add context padding
            bw, bh = x2 - x1, y2 - y1
            pad_px_w = int(bw * context_pad)
            pad_px_h = int(bh * context_pad)
            x1 = max(0, x1 - pad_px_w)
            y1 = max(0, y1 - pad_px_h)
            x2 = min(w, x2 + pad_px_w)
            y2 = min(h, y2 + pad_px_h)

            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            crop_resized = letterbox_crop(crop, size)

            phash = perceptual_hash(crop_resized)
            if phash in seen_hashes:
                continue
            seen_hashes.add(phash)

            out_name = f"{img_path.stem}_cap_crop{count}.jpg"
            cv2.imwrite(
                str(output_dir / out_name),
                crop_resized,
                [cv2.IMWRITE_JPEG_QUALITY, 95],
            )
            count += 1

    return count


def main():
    parser = argparse.ArgumentParser(
        description="Extract cap crops for classification training"
    )
    parser.add_argument("--mode", choices=["labels", "model"], default="labels",
                        help="Extraction mode: from YOLO labels or from a model")
    parser.add_argument("--images", required=True,
                        help="Path to images folder")
    parser.add_argument("--labels", default="",
                        help="Path to YOLO labels folder (mode=labels)")
    parser.add_argument("--weights", default="",
                        help="Path to detector .pt weights (mode=model)")
    parser.add_argument("--output", default="data/caps/unsorted",
                        help="Output folder for cropped caps")
    parser.add_argument("--cap-class-ids", default="0,1",
                        help="Comma-separated class IDs to crop (mode=labels)")
    parser.add_argument("--cap-class-name", default="cap",
                        help="Class name to crop (mode=model)")
    parser.add_argument("--size", type=int, default=224,
                        help="Output crop size (letterbox, default: 224)")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Confidence threshold (mode=model)")
    parser.add_argument("--context-pad", type=float, default=0.1,
                        help="Context padding ratio around bbox (default: 0.1)")
    args = parser.parse_args()

    images_dir = Path(args.images)
    output_dir = Path(args.output)
    assert images_dir.exists(), f"Images folder not found: {images_dir}"

    print(f"\n{'='*55}")
    print("  Cap Crop Extractor")
    print(f"{'='*55}")
    print(f"  Mode    : {args.mode}")
    print(f"  Images  : {images_dir}")
    print(f"  Output  : {output_dir}")
    print(f"  Size    : {args.size}×{args.size} (letterbox)")
    print()

    if args.mode == "labels":
        labels_dir = Path(args.labels)
        assert labels_dir.exists(), f"Labels folder not found: {labels_dir}"
        cap_ids = {int(x.strip()) for x in args.cap_class_ids.split(",") if x.strip()}
        count = extract_from_labels(
            images_dir, labels_dir, output_dir, cap_ids, args.size, args.context_pad
        )
    else:
        weights_path = Path(args.weights)
        assert weights_path.exists(), f"Weights not found: {weights_path}"
        count = extract_from_model(
            images_dir, weights_path, output_dir,
            args.cap_class_name, args.size, args.conf, args.context_pad,
        )

    print(f"  Extracted {count} cap crops → {output_dir.resolve()}")
    print(f"\n  → Next: manually sort into good_cap/ and defective_cap/ subfolders")
    print(f"  → Then: train classifier with train_cap_classifier.py\n")


if __name__ == "__main__":
    main()
