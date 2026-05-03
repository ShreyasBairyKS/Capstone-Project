"""
scripts/fill_level_extractor.py
================================
Extract bottle crops from waterfill1 dataset and auto-sort into fill level folders.

Dataset classes (waterfill1):
    0: empty              → underfill/
    1: full_water_level   → overfill/
    2: half_water_level   → underfill/
    3: three_quarters_level → normal/

Target 3-class structure:
    data/fill_level/
        underfill/    ← empty + half
        normal/       ← three_quarters
        overfill/     ← full

Usage (run for train, valid, test):
    python scripts/fill_level_extractor.py --images "E:\P-25 Vision Food ai\dataset\Beverages\waterfill1\train\images" --output data/fill_level --size 224

    python scripts/fill_level_extractor.py --images "E:\P-25 Vision Food ai\dataset\Beverages\waterfill1\valid\images" --output data/fill_level --size 224

    python scripts/fill_level_extractor.py --images "E:\P-25 Vision Food ai\dataset\Beverages\waterfill1\test\images" --output data/fill_level --size 224

    # Then train:
    python training/train_fill_classifier.py --data-root data/fill_level --auto-split --epochs 50 --device 0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

# ── Default class mapping ──
# waterfill1: 0=empty, 1=full_water_level, 2=half_water_level, 3=three_quarters_level
DEFAULT_CLASS_MAP = "0:underfill,1:overfill,2:underfill,3:normal"


def letterbox_crop(crop: np.ndarray, size: int) -> np.ndarray:
    """Resize crop preserving aspect ratio with gray padding."""
    h, w = crop.shape[:2]
    ratio = min(size / h, size / w)
    new_h, new_w = int(round(h * ratio)), int(round(w * ratio))
    resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_top  = (size - new_h) // 2
    pad_left = (size - new_w) // 2
    padded = cv2.copyMakeBorder(
        resized,
        pad_top, size - new_h - pad_top,
        pad_left, size - new_w - pad_left,
        cv2.BORDER_CONSTANT, value=(114, 114, 114),
    )
    return padded


def perceptual_hash(img: np.ndarray, hash_size: int = 8) -> str:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    resized = cv2.resize(gray, (hash_size, hash_size), interpolation=cv2.INTER_AREA)
    mean = resized.mean()
    bits = (resized > mean).flatten()
    h = 0
    for bit in bits:
        h = (h << 1) | int(bit)
    return f"{h:016x}"


def parse_class_map(s: str) -> dict[int, str]:
    mapping = {}
    for pair in s.split(","):
        pair = pair.strip()
        if ":" not in pair:
            continue
        cls_id, folder = pair.split(":", 1)
        mapping[int(cls_id.strip())] = folder.strip()
    return mapping


def extract(
    images_dir: Path,
    labels_dir: Path,
    output_dir: Path,
    class_map: dict[int, str],
    size: int,
    context_pad: float,
) -> dict[str, int]:
    """
    Extract whole-bottle crops from YOLO labels and sort by fill class.

    NOTE: These labels annotate the entire bottle (or liquid region).
    We crop the bounding box from the image → resize → save to class folder.
    """
    # Create output folders
    for folder in set(class_map.values()):
        (output_dir / folder).mkdir(parents=True, exist_ok=True)

    seen_hashes: set[str] = set()
    counts: dict[str, int] = {}
    skipped = 0

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

        label_text = lbl_path.read_text(encoding="utf-8").strip()
        if not label_text:
            # Empty label = no annotation; skip
            skipped += 1
            continue

        for line in label_text.splitlines():
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            cls_id = int(float(parts[0]))
            if cls_id not in class_map:
                skipped += 1
                continue

            folder = class_map[cls_id]
            cx, cy, bw, bh = map(float, parts[1:5])

            # Add context padding around the bbox
            pad_w = bw * context_pad
            pad_h = bh * context_pad
            x1 = max(0, int((cx - bw/2 - pad_w) * w))
            y1 = max(0, int((cy - bh/2 - pad_h) * h))
            x2 = min(w, int((cx + bw/2 + pad_w) * w))
            y2 = min(h, int((cy + bh/2 + pad_h) * h))

            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            crop_resized = letterbox_crop(crop, size)

            # Deduplication
            phash = perceptual_hash(crop_resized)
            if phash in seen_hashes:
                continue
            seen_hashes.add(phash)

            n = counts.get(folder, 0)
            out_name = f"{img_path.stem}_cls{cls_id}_{n}.jpg"
            cv2.imwrite(
                str(output_dir / folder / out_name),
                crop_resized,
                [cv2.IMWRITE_JPEG_QUALITY, 95],
            )
            counts[folder] = n + 1

    return counts


def main():
    parser = argparse.ArgumentParser(
        description="Extract bottle crops for fill level classification"
    )
    parser.add_argument("--images", required=True, help="Path to images folder")
    parser.add_argument("--labels", default="",
                        help="Path to labels folder (auto-detected if empty)")
    parser.add_argument("--output", default="data/fill_level",
                        help="Output root folder")
    parser.add_argument("--class-map", default=DEFAULT_CLASS_MAP,
                        help="Map class IDs to folders. "
                             "Default: 0(empty)→underfill, 1(full)→overfill, "
                             "2(half)→underfill, 3(three_quarters)→normal")
    parser.add_argument("--size", type=int, default=224,
                        help="Output crop size (letterbox, default: 224)")
    parser.add_argument("--context-pad", type=float, default=0.05,
                        help="Context padding around bbox (default: 0.05 = 5%%)")
    args = parser.parse_args()

    images_dir = Path(args.images)
    labels_dir = Path(args.labels) if args.labels else images_dir.parent.parent / "labels" / images_dir.parent.name
    # Fallback: sibling labels folder
    if not labels_dir.exists():
        labels_dir = images_dir.parent / "labels"
    if not labels_dir.exists():
        # Try replacing 'images' with 'labels' in the path
        labels_dir = Path(str(images_dir).replace("images", "labels"))

    output_dir = Path(args.output)
    class_map = parse_class_map(args.class_map)

    assert images_dir.exists(), f"Images not found: {images_dir}"
    assert labels_dir.exists(), f"Labels not found: {labels_dir}\nPass --labels explicitly."

    print(f"\n{'='*55}")
    print("  Fill Level Crop Extractor")
    print(f"{'='*55}")
    print(f"  Images  : {images_dir}")
    print(f"  Labels  : {labels_dir}")
    print(f"  Output  : {output_dir}")
    print(f"  Size    : {args.size}x{args.size}")
    print(f"\n  Class mapping:")
    for cls_id, folder in sorted(class_map.items()):
        print(f"    Class {cls_id} → {folder}/")
    print()

    counts = extract(images_dir, labels_dir, output_dir, class_map, args.size, args.context_pad)

    total = sum(counts.values())
    print(f"  ── Results ──")
    for folder, n in sorted(counts.items()):
        print(f"    {folder:>12}: {n} crops")
    print(f"    {'TOTAL':>12}: {total} crops")

    print(f"\n  Output structure:")
    print(f"    {output_dir}/")
    for folder in sorted(set(class_map.values())):
        n = counts.get(folder, 0)
        print(f"      {folder}/  ({n} images)")

    print(f"\n  → NEXT: Train fill level classifier:")
    print(f"    python training/train_fill_classifier.py \\")
    print(f"      --data-root {output_dir} --auto-split --epochs 50 --device 0 --export")
    print()


if __name__ == "__main__":
    main()
