# Dataset Validator — Check dataset quality before training
"""
scripts/dataset_validator.py
==============================
Validates dataset quality before training.

Checks:
    1. Class balance (warns if >3:1 ratio)
    2. Duplicate/near-duplicate images (perceptual hashing)
    3. Annotation format (YOLO .txt validation)
    4. Image resolution distribution
    5. Corrupt images

Usage:
    # Validate detection dataset:
    python scripts/dataset_validator.py \
        --images dataset/bottle_cap_det/train/images \
        --labels dataset/bottle_cap_det/train/labels \
        --mode detection

    # Validate classification dataset:
    python scripts/dataset_validator.py \
        --images data/caps/train \
        --mode classification
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def perceptual_hash(img: np.ndarray, hash_size: int = 8) -> str:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    resized = cv2.resize(gray, (hash_size, hash_size), interpolation=cv2.INTER_AREA)
    mean = resized.mean()
    bits = (resized > mean).flatten()
    hash_int = 0
    for bit in bits:
        hash_int = (hash_int << 1) | int(bit)
    return f"{hash_int:016x}"


def hamming_distance(h1: str, h2: str) -> int:
    v1 = int(h1, 16)
    v2 = int(h2, 16)
    return bin(v1 ^ v2).count("1")


def validate_detection(images_dir: Path, labels_dir: Path):
    print(f"\n  Mode: Detection")
    print(f"  Images: {images_dir}")
    print(f"  Labels: {labels_dir}")

    image_files = sorted(p for p in images_dir.glob("*.*") if p.suffix.lower() in IMAGE_EXTENSIONS)

    if not image_files:
        print(f"  ❌ No images found!")
        return

    # Check basics
    class_counter = Counter()
    corrupt = []
    missing_labels = []
    bad_labels = []
    resolutions = []
    hashes = {}
    duplicates = []

    for img_path in image_files:
        # Check image readability
        img = cv2.imread(str(img_path))
        if img is None:
            corrupt.append(img_path.name)
            continue

        h, w = img.shape[:2]
        resolutions.append((w, h))

        # Dedup check
        phash = perceptual_hash(img)
        if phash in hashes:
            duplicates.append((img_path.name, hashes[phash]))
        else:
            hashes[phash] = img_path.name

        # Check label
        lbl_path = labels_dir / f"{img_path.stem}.txt"
        if not lbl_path.exists():
            missing_labels.append(img_path.name)
            continue

        for line_num, line in enumerate(lbl_path.read_text().strip().splitlines(), 1):
            parts = line.strip().split()
            if len(parts) < 5:
                bad_labels.append(f"{lbl_path.name}:{line_num}")
                continue
            try:
                cls_id = int(float(parts[0]))
                vals = [float(x) for x in parts[1:5]]
                class_counter[cls_id] += 1
                # Validate ranges
                if any(v < 0 or v > 1 for v in vals):
                    bad_labels.append(f"{lbl_path.name}:{line_num} (out of range)")
            except ValueError:
                bad_labels.append(f"{lbl_path.name}:{line_num} (parse error)")

    # Report
    print(f"\n  ── Dataset Summary ──")
    print(f"  Total images      : {len(image_files)}")
    print(f"  Corrupt images    : {len(corrupt)}")
    print(f"  Missing labels    : {len(missing_labels)}")
    print(f"  Bad label lines   : {len(bad_labels)}")
    print(f"  Duplicate images  : {len(duplicates)}")

    print(f"\n  ── Class Distribution ──")
    total_annots = sum(class_counter.values())
    for cls_id, count in sorted(class_counter.items()):
        pct = 100 * count / max(total_annots, 1)
        bar = "█" * int(pct / 2)
        print(f"  Class {cls_id}: {count:>6} ({pct:>5.1f}%) {bar}")

    # Balance check
    if len(class_counter) >= 2:
        counts = list(class_counter.values())
        ratio = max(counts) / max(min(counts), 1)
        if ratio > 3:
            print(f"\n  ⚠️  Class imbalance detected! Ratio: {ratio:.1f}:1")
            print(f"     → Consider augmenting the minority class")

    # Resolution stats
    if resolutions:
        ws, hs = zip(*resolutions)
        print(f"\n  ── Resolution Stats ──")
        print(f"  Width  : min={min(ws)}, max={max(ws)}, avg={np.mean(ws):.0f}")
        print(f"  Height : min={min(hs)}, max={max(hs)}, avg={np.mean(hs):.0f}")

    if duplicates:
        print(f"\n  ── Duplicates (first 10) ──")
        for img1, img2 in duplicates[:10]:
            print(f"    {img1}  ≈  {img2}")

    if corrupt:
        print(f"\n  ── Corrupt Images ──")
        for name in corrupt[:10]:
            print(f"    {name}")


def validate_classification(images_dir: Path):
    print(f"\n  Mode: Classification")
    print(f"  Root: {images_dir}")

    total = 0
    class_counts = {}

    for subdir in sorted(images_dir.iterdir()):
        if not subdir.is_dir():
            continue
        imgs = [p for p in subdir.glob("*.*") if p.suffix.lower() in IMAGE_EXTENSIONS]
        class_counts[subdir.name] = len(imgs)
        total += len(imgs)

    if not class_counts:
        print(f"  ❌ No class subfolders found!")
        return

    print(f"\n  ── Class Distribution ──")
    print(f"  Total images: {total}")
    for cls_name, count in sorted(class_counts.items()):
        pct = 100 * count / max(total, 1)
        bar = "█" * int(pct / 2)
        print(f"  {cls_name:<20}: {count:>6} ({pct:>5.1f}%) {bar}")

    counts = list(class_counts.values())
    if len(counts) >= 2:
        ratio = max(counts) / max(min(counts), 1)
        if ratio > 3:
            print(f"\n  ⚠️  Class imbalance! Ratio: {ratio:.1f}:1")


def main():
    parser = argparse.ArgumentParser(description="Validate dataset quality")
    parser.add_argument("--images", required=True, help="Images folder")
    parser.add_argument("--labels", default="", help="Labels folder (detection mode)")
    parser.add_argument("--mode", choices=["detection", "classification"],
                        default="detection")
    args = parser.parse_args()

    print(f"\n{'='*55}")
    print("  Dataset Validator")
    print(f"{'='*55}")

    if args.mode == "detection":
        labels_dir = Path(args.labels) if args.labels else Path(args.images).parent / "labels"
        validate_detection(Path(args.images), labels_dir)
    else:
        validate_classification(Path(args.images))

    print()


if __name__ == "__main__":
    main()
