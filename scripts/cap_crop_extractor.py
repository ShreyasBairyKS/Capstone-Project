"""
scripts/cap_crop_extractor.py
==============================
Extract cap crops from YOLO labels and AUTO-SORT into classifier folders.

Your Roboflow dataset classes:
    0: Broken Cap   → defective_cap/
    1: Broken Ring   → defective_cap/
    2: Good Cap      → good_cap/
    3: Loose Cap     → defective_cap/
    4: No Cap        → SKIP (nothing to crop)

Usage (AUTO-SORT — no manual work needed):
    python scripts/cap_crop_extractor.py \
        --images dataset/Beverages/bottleDefect.v1-first.yolov11-cap/train/images \
        --labels dataset/Beverages/bottleDefect.v1-first.yolov11-cap/train/labels \
        --output data/caps \
        --class-map "0:defective_cap,1:defective_cap,2:good_cap,3:defective_cap" \
        --size 224

    # Then train directly:
    python training/train_cap_classifier.py \
        --data-root data/caps --auto-split --epochs 50 --device 0
"""

from __future__ import annotations

import argparse
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
        cv2.BORDER_CONSTANT, value=(114, 114, 114),
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


def parse_class_map(class_map_str: str) -> dict[int, str]:
    """Parse class map string like '0:defective_cap,1:defective_cap,2:good_cap'."""
    mapping = {}
    for pair in class_map_str.split(","):
        pair = pair.strip()
        if ":" not in pair:
            continue
        cls_id, folder = pair.split(":", 1)
        mapping[int(cls_id.strip())] = folder.strip()
    return mapping


def extract_and_sort(
    images_dir: Path,
    labels_dir: Path,
    output_dir: Path,
    class_map: dict[int, str],
    size: int,
    context_pad: float,
) -> dict[str, int]:
    """Extract cap crops and auto-sort into class folders."""
    # Create output folders
    for folder_name in set(class_map.values()):
        (output_dir / folder_name).mkdir(parents=True, exist_ok=True)

    seen_hashes: set[str] = set()
    counts: dict[str, int] = {}
    dupes = 0
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

        for line in lbl_path.read_text(encoding="utf-8").strip().splitlines():
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            cls_id = int(float(parts[0]))

            # Skip classes not in the map (e.g., No Cap = class 4)
            if cls_id not in class_map:
                skipped += 1
                continue

            folder_name = class_map[cls_id]
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

            crop_resized = letterbox_crop(crop, size)

            # Deduplication
            phash = perceptual_hash(crop_resized)
            if phash in seen_hashes:
                dupes += 1
                continue
            seen_hashes.add(phash)

            # Save to the right folder
            crop_count = counts.get(folder_name, 0)
            out_name = f"{img_path.stem}_cls{cls_id}_{crop_count}.jpg"
            out_path = output_dir / folder_name / out_name
            cv2.imwrite(str(out_path), crop_resized, [cv2.IMWRITE_JPEG_QUALITY, 95])
            counts[folder_name] = crop_count + 1

    return counts


def main():
    parser = argparse.ArgumentParser(
        description="Extract and auto-sort cap crops for classifier training"
    )
    parser.add_argument("--images", required=True,
                        help="Path to images folder")
    parser.add_argument("--labels", default="",
                        help="Path to YOLO labels folder")
    parser.add_argument("--output", default="data/caps",
                        help="Output root folder (subfolders created automatically)")
    parser.add_argument("--class-map",
                        default="0:defective_cap,1:defective_cap,2:good_cap,3:defective_cap",
                        help="Map class IDs to folders. Format: 'id:folder,id:folder,...' "
                             "Default maps Broken Cap/Ring/Loose→defective, Good→good, No Cap→skip")
    parser.add_argument("--size", type=int, default=224,
                        help="Output crop size (letterbox, default: 224)")
    parser.add_argument("--context-pad", type=float, default=0.1,
                        help="Context padding ratio around bbox (default: 0.1)")
    args = parser.parse_args()

    images_dir = Path(args.images)
    labels_dir = Path(args.labels) if args.labels else images_dir.parent / "labels"
    output_dir = Path(args.output)

    assert images_dir.exists(), f"Images folder not found: {images_dir}"
    assert labels_dir.exists(), f"Labels folder not found: {labels_dir}"

    class_map = parse_class_map(args.class_map)

    print(f"\n{'='*55}")
    print("  Cap Crop Extractor (Auto-Sort)")
    print(f"{'='*55}")
    print(f"  Images  : {images_dir}")
    print(f"  Labels  : {labels_dir}")
    print(f"  Output  : {output_dir}")
    print(f"  Size    : {args.size}x{args.size} (letterbox)")
    print(f"\n  Class mapping:")
    for cls_id, folder in sorted(class_map.items()):
        print(f"    Class {cls_id} → {folder}/")
    print(f"    (unlisted IDs → SKIP)")
    print()

    counts = extract_and_sort(
        images_dir, labels_dir, output_dir, class_map, args.size, args.context_pad
    )

    total = sum(counts.values())
    print(f"  ── Results ──")
    for folder, n in sorted(counts.items()):
        print(f"    {folder:>20}: {n} crops")
    print(f"    {'TOTAL':>20}: {total} crops")

    print(f"\n  Output structure:")
    print(f"    {output_dir}/")
    for folder in sorted(set(class_map.values())):
        n = counts.get(folder, 0)
        print(f"      {folder}/  ({n} images)")

    print(f"\n  → NEXT: Train classifier:")
    print(f"    python training/train_cap_classifier.py \\")
    print(f"      --data-root {output_dir} \\")
    print(f"      --auto-split --epochs 50 --batch 64 --device 0 --export")
    print()


if __name__ == "__main__":
    main()
