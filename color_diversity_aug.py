"""
color_diversity_aug.py
======================
Generates synthetically color-varied training images from an existing
single-color bottle dataset.

Simulates real-world bottle/cap color diversity:
    Bottle colors : transparent, amber, green, blue, brown
    Cap colors    : red, blue, black, yellow, silver, green

Strategy:
    1. Detect the cap region using YOLO label bbox coordinates
    2. Apply targeted hue/saturation shifts to cap region specifically
    3. Apply global bottle tint transforms
    4. Save augmented image/label pairs

This encourages the model to learn defect structure (shape, edge, deformation)
rather than overfitting to narrow color patterns.

Run:
    python color_diversity_aug.py \
        --train-images ./data/images/train \
        --train-labels ./data/labels/train \
        --output-dir   ./data_augmented \
        --variants     6 \
        --defect-only
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

# (hue_target, saturation_boost, label)
CAP_COLOR_VARIANTS: list[tuple[int, int, str]] = [
    (0, 180, "red_cap"),
    (105, 180, "blue_cap"),
    (0, 0, "black_cap"),
    (25, 160, "yellow_cap"),
    (60, 140, "green_cap"),
    (0, 30, "silver_cap"),
]

# (R, G, B), label
BOTTLE_TINTS: list[tuple[tuple[float, float, float], str]] = [
    ((0.95, 0.85, 0.65), "amber"),
    ((0.75, 0.95, 0.75), "green"),
    ((0.75, 0.85, 0.95), "blue"),
    ((0.95, 0.95, 0.95), "clear"),
    ((0.65, 0.55, 0.45), "dark_brown"),
]


def yolo_bbox_to_pixel(
    bbox_norm: list[float],
    img_h: int,
    img_w: int,
) -> tuple[int, int, int, int]:
    """Convert YOLO normalized [cx, cy, w, h] to pixel [x1, y1, x2, y2]."""
    cx, cy, w, h = bbox_norm
    x1 = int((cx - w / 2.0) * img_w)
    y1 = int((cy - h / 2.0) * img_h)
    x2 = int((cx + w / 2.0) * img_w)
    y2 = int((cy + h / 2.0) * img_h)
    return max(0, x1), max(0, y1), min(img_w, x2), min(img_h, y2)


def load_labels(label_path: Path) -> list[list[float]]:
    """Return list of [cls, cx, cy, w, h] rows from a YOLO .txt file."""
    if not label_path.exists() or label_path.stat().st_size == 0:
        return []

    labels: list[list[float]] = []
    for line in label_path.read_text(encoding="utf-8").strip().splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        labels.append([int(parts[0]), *map(float, parts[1:])])
    return labels


def recolor_region(
    image: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    hue_target: int,
    sat_value: int,
) -> np.ndarray:
    """Recolor a rectangular region while preserving luminance structure."""
    img = image.copy()
    region = img[y1:y2, x1:x2]
    if region.size == 0:
        return img

    hsv_region = cv2.cvtColor(region, cv2.COLOR_BGR2HSV).astype(np.float32)

    if sat_value == 0:
        # Black cap simulation.
        hsv_region[:, :, 1] = 0
        hsv_region[:, :, 2] *= 0.25
    else:
        hsv_region[:, :, 0] = hue_target
        hsv_region[:, :, 1] = np.clip(
            hsv_region[:, :, 1] * 0.3 + sat_value * 0.7,
            0,
            255,
        )

    hsv_region = np.clip(hsv_region, 0, 255).astype(np.uint8)
    img[y1:y2, x1:x2] = cv2.cvtColor(hsv_region, cv2.COLOR_HSV2BGR)
    return img


def apply_bottle_tint(image: np.ndarray, tint_rgb: tuple[float, float, float]) -> np.ndarray:
    """Apply a multiplicative tint to simulate colored glass/PET bottles."""
    img = image.astype(np.float32)
    # OpenCV is BGR, tint_rgb is (R, G, B).
    r, g, b = tint_rgb
    img[:, :, 0] *= b
    img[:, :, 1] *= g
    img[:, :, 2] *= r
    return np.clip(img, 0, 255).astype(np.uint8)


def expand_bbox_to_cap_area(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    expand: float = 0.3,
) -> tuple[int, int, int, int]:
    """Expand bbox upward to capture cap context around tightly-labeled defects."""
    bh = y2 - y1
    y1_exp = max(0, y1 - int(bh * expand))
    return x1, y1_exp, x2, y2


def image_has_defect(labels: list[list[float]], defect_class_ids: set[int]) -> bool:
    """Return True if any label belongs to the configured defect class ids."""
    if not labels:
        return False
    return any(int(lbl[0]) in defect_class_ids for lbl in labels)


def generate_variants(
    image: np.ndarray,
    labels: list[list[float]],
    n_variants: int,
    rng: random.Random,
) -> list[tuple[np.ndarray, list[list[float]]]]:
    """Generate n color-diverse versions of image while keeping labels unchanged."""
    if n_variants <= 0:
        return []

    img_h, img_w = image.shape[:2]
    results: list[tuple[np.ndarray, list[list[float]]]] = []

    cap_variants = rng.sample(CAP_COLOR_VARIANTS, min(n_variants, len(CAP_COLOR_VARIANTS)))
    bottle_tints = rng.choices(BOTTLE_TINTS, k=n_variants)

    for i in range(n_variants):
        aug = image.copy()

        # 1) Global bottle tint.
        tint_rgb, _ = bottle_tints[i]
        aug = apply_bottle_tint(aug, tint_rgb)

        # 2) Cap recolor from labels.
        if labels and i < len(cap_variants):
            hue, sat, _ = cap_variants[i]
            for lbl in labels:
                _, cx, cy, bw, bh = lbl
                x1, y1, x2, y2 = yolo_bbox_to_pixel([cx, cy, bw, bh], img_h, img_w)
                x1, y1, x2, y2 = expand_bbox_to_cap_area(x1, y1, x2, y2)
                aug = recolor_region(aug, x1, y1, x2, y2, hue, sat)

        results.append((aug, labels))

    return results


def serialize_labels(labels: list[list[float]]) -> str:
    """Serialize labels back to YOLO txt format."""
    return "\n".join(
        f"{int(row[0])} {row[1]:.6f} {row[2]:.6f} {row[3]:.6f} {row[4]:.6f}"
        for row in labels
    )


def parse_class_ids(raw: str) -> set[int]:
    """Parse comma-separated class ids into a set of ints."""
    parsed: set[int] = set()
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        parsed.add(int(token))
    return parsed


def parse_args() -> argparse.Namespace:
    """Build and parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate color-diverse train split augmentations for VisionFood QAI."
    )
    parser.add_argument("--train-images", default="./data/images/train")
    parser.add_argument("--train-labels", default="./data/labels/train")
    parser.add_argument(
        "--output-dir",
        default="./data_augmented",
        help="Output dataset root. Original train pairs are copied too.",
    )
    parser.add_argument(
        "--variants",
        default=6,
        type=int,
        help="Color variants per image (max 6).",
    )
    parser.add_argument(
        "--defect-only",
        action="store_true",
        help="Only augment images containing defect labels.",
    )
    parser.add_argument(
        "--defect-class-ids",
        default="0",
        help="Comma-separated class ids treated as defective in --defect-only mode.",
    )
    parser.add_argument(
        "--seed",
        default=42,
        type=int,
        help="Random seed for reproducible variant sampling.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    src_imgs = Path(args.train_images)
    src_lbls = Path(args.train_labels)
    dst_imgs = Path(args.output_dir) / "images" / "train"
    dst_lbls = Path(args.output_dir) / "labels" / "train"

    if not src_imgs.exists():
        raise SystemExit(f"train-images path not found: {src_imgs}")
    if not src_lbls.exists():
        raise SystemExit(f"train-labels path not found: {src_lbls}")
    if args.variants <= 0:
        raise SystemExit("--variants must be > 0")

    dst_imgs.mkdir(parents=True, exist_ok=True)
    dst_lbls.mkdir(parents=True, exist_ok=True)

    n_variants = min(args.variants, len(CAP_COLOR_VARIANTS))
    defect_class_ids = parse_class_ids(args.defect_class_ids)
    rng = random.Random(args.seed)

    print("\n" + "=" * 58)
    print("  VisionFood QAI - Color Diversity Augmentation")
    print("=" * 58)
    print(f"  Source train images : {src_imgs}")
    print(f"  Source train labels : {src_lbls}")
    print(f"  Output root         : {Path(args.output_dir)}")
    print(f"  Variants            : {n_variants} per image")
    print(f"  Defect only         : {args.defect_only}")
    if args.defect_only:
        print(f"  Defect class ids    : {sorted(defect_class_ids)}")
    print(f"  Seed                : {args.seed}")
    print()

    image_files = sorted(
        p for p in src_imgs.glob("*.*") if p.suffix.lower() in IMAGE_EXTENSIONS
    )

    copied = 0
    generated = 0
    skipped = 0

    for img_path in image_files:
        lbl_path = src_lbls / f"{img_path.stem}.txt"
        labels = load_labels(lbl_path)

        # Always copy original train pair into output.
        shutil.copy2(img_path, dst_imgs / img_path.name)
        dst_label_path = dst_lbls / lbl_path.name
        if lbl_path.exists():
            shutil.copy2(lbl_path, dst_label_path)
        else:
            dst_label_path.touch()
        copied += 1

        # Optionally augment only defect-class images.
        if args.defect_only and not image_has_defect(labels, defect_class_ids):
            skipped += 1
            continue

        image = cv2.imread(str(img_path))
        if image is None:
            continue

        variants = generate_variants(image, labels, n_variants, rng)
        for i, (aug_img, aug_labels) in enumerate(variants):
            stem = f"{img_path.stem}_col{i}"
            out_img = dst_imgs / f"{stem}.jpg"
            out_lbl = dst_lbls / f"{stem}.txt"
            cv2.imwrite(str(out_img), aug_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            out_lbl.write_text(serialize_labels(aug_labels), encoding="utf-8")
            generated += 1

    total = copied + generated
    print(f"  Original train images copied : {copied}")
    print(f"  Color variants created       : {generated}")
    print(f"  Total train images           : {total}")
    print(f"  Skipped (non-defect images)  : {skipped}")

    print("\nAugmented train split is ready.")
    print(f"Dataset root: {Path(args.output_dir).resolve()}")
    print("\nNext step: copy val/test splits unchanged into output-dir and retrain.")
    print("Example:")
    print(f"  - Copy source images/val  to {Path(args.output_dir) / 'images' / 'val'}")
    print(f"  - Copy source images/test to {Path(args.output_dir) / 'images' / 'test'}")
    print(f"  - Copy source labels/val  to {Path(args.output_dir) / 'labels' / 'val'}")
    print(f"  - Copy source labels/test to {Path(args.output_dir) / 'labels' / 'test'}")
    print(f"  - Copy source data.yaml   to {Path(args.output_dir) / 'data.yaml'}")


if __name__ == "__main__":
    main()
