"""
scripts/merge_datasets.py
===========================
Merge multiple Roboflow YOLO datasets with DIFFERENT class names into
one unified dataset for bottle+cap detection.

Scans all subdirectories under a root folder, reads each data.yaml,
auto-maps class names to:
    0: bottle
    1: cap

Any class name containing "bottle" → mapped to 0 (bottle)
Any class name containing "cap"    → mapped to 1 (cap)
Everything else is SKIPPED (not included).

Usage:
    python scripts/merge_datasets.py \
        --input  dataset/Beverages/hybrid_bottle_cap_new \
        --output dataset/bottle_cap_merged

After merging, train with:
    python training/train_detector.py \
        --data dataset/bottle_cap_merged/data.yaml \
        --model yolov8s --epochs 150 --profile a5000-balanced --device 0
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

import yaml

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".PNG", ".JPEG"}

# ─────────────────────────────────────────────────────────────────────────────
# CLASS NAME AUTO-MAPPING
# ─────────────────────────────────────────────────────────────────────────────
# Any class name matching these keywords → mapped to unified ID
# This handles all common naming variations across Roboflow datasets

BOTTLE_KEYWORDS = ["bottle", "Bottle", "BOTTLE", "pet", "container"]
CAP_KEYWORDS = [
    "cap", "Cap", "CAP", "lid", "Lid",
    # Explicit multi-word names from common Roboflow datasets
    "bottle cap", "bottle_cap", "Bottle Cap",
    "Good Cap", "good cap", "good_cap", "goodCap",
    "Broken Cap", "broken cap", "broken_cap",
    "Loose Cap", "loose cap", "loose_cap",
    "No Cap", "no cap", "noCap", "no_cap",
    "Broken Ring", "broken ring", "broken_ring",
    # Other variations
    "defectcap", "defectCap", "defect_cap", "defect-cap",
    "badcap", "badCap", "bad_cap", "bad-cap",
    "damaged_cap", "missing_cap",
    "closed", "open",
    "sealed", "unsealed",
    "crown", "screw_cap", "screwcap",
]

UNIFIED_NAMES = {0: "bottle", 1: "cap"}


def classify_class_name(name: str) -> int | None:
    """Map a class name to unified ID: 0=bottle, 1=cap, None=skip."""
    name_lower = name.lower().strip()

    # Check cap first (more specific — "bottle_cap" should map to cap, not bottle)
    for kw in CAP_KEYWORDS:
        if kw.lower() in name_lower:
            return 1

    for kw in BOTTLE_KEYWORDS:
        if kw.lower() in name_lower:
            return 0

    return None  # unknown class — skip


def find_datasets(root: Path) -> list[Path]:
    """Find all data.yaml files under the root directory."""
    yamls = sorted(root.rglob("data.yaml"))
    if not yamls:
        # Also check for .yml
        yamls = sorted(root.rglob("data.yml"))
    return yamls


def parse_class_names(data_yaml: Path) -> dict[int, str]:
    """Read class names from a YOLO data.yaml."""
    with open(data_yaml, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    names = cfg.get("names", {})
    if isinstance(names, list):
        return {i: n for i, n in enumerate(names)}
    elif isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    return {}


def resolve_split_dir(data_yaml: Path, cfg: dict, split_name: str) -> tuple[Path, Path] | None:
    """Resolve image and label directories for a split."""
    split_val = cfg.get(split_name)
    if not split_val:
        return None

    # Try relative to data.yaml parent
    img_dir = (data_yaml.parent / split_val).resolve()
    if not img_dir.exists():
        # Try relative to "path" in yaml
        root_path = cfg.get("path", "")
        if root_path:
            img_dir = (Path(root_path) / split_val).resolve()

    if not img_dir.exists():
        return None

    # Labels directory: images/ → labels/
    lbl_dir = Path(str(img_dir).replace("/images", "/labels").replace("\\images", "\\labels"))
    if not lbl_dir.exists():
        # Try sibling
        lbl_dir = img_dir.parent / "labels"

    return img_dir, lbl_dir


def obb_to_xywh(coords: list[float]) -> tuple[float, float, float, float]:
    """
    Convert oriented bounding box (4 corner points, 8 values) to
    axis-aligned bounding box in YOLO format (cx, cy, w, h).

    OBB format: x1 y1 x2 y2 x3 y3 x4 y4  (normalized 0-1)
    Output:     cx cy w h                   (normalized 0-1)
    """
    xs = [coords[i] for i in range(0, 8, 2)]  # x1, x2, x3, x4
    ys = [coords[i] for i in range(1, 8, 2)]  # y1, y2, y3, y4
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    cx = (x_min + x_max) / 2.0
    cy = (y_min + y_max) / 2.0
    w = x_max - x_min
    h = y_max - y_min
    return cx, cy, w, h


def remap_label_file(
    src_label: Path,
    dst_label: Path,
    old_to_new: dict[int, int],
):
    """
    Read a YOLO label file, remap class IDs, write to destination.

    Handles two formats:
      - Standard:  class cx cy w h            (5 values)
      - OBB:       class x1 y1 x2 y2 x3 y3 x4 y4  (9 values)
        → Converted to axis-aligned bounding box automatically.
    """
    if not src_label.exists():
        # Empty label = no objects (background image — still useful!)
        dst_label.write_text("", encoding="utf-8")
        return 0

    lines = src_label.read_text(encoding="utf-8").strip().splitlines()
    new_lines = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        try:
            old_cls = int(float(parts[0]))
        except ValueError:
            continue

        if old_cls not in old_to_new:
            continue  # skip unknown classes

        new_cls = old_to_new[old_cls]

        if len(parts) == 9:
            # OBB format: class x1 y1 x2 y2 x3 y3 x4 y4
            # Convert to axis-aligned: class cx cy w h
            try:
                coords = [float(v) for v in parts[1:9]]
                cx, cy, w, h = obb_to_xywh(coords)
                # Clamp to valid range
                cx = max(0.0, min(1.0, cx))
                cy = max(0.0, min(1.0, cy))
                w = max(0.001, min(1.0, w))
                h = max(0.001, min(1.0, h))
                new_lines.append(f"{new_cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            except (ValueError, IndexError):
                continue
        elif len(parts) >= 5:
            # Standard format: class cx cy w h
            parts[0] = str(new_cls)
            # Keep only the first 5 values (class + bbox)
            new_lines.append(" ".join(parts[:5]))

    dst_label.write_text("\n".join(new_lines) + ("\n" if new_lines else ""),
                         encoding="utf-8")
    return len(new_lines)


def merge_one_dataset(
    data_yaml: Path,
    output_dir: Path,
    dataset_prefix: str,
    stats: dict,
):
    """Merge one dataset into the unified output."""
    with open(data_yaml, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    class_names = parse_class_names(data_yaml)
    if not class_names:
        print(f"    ⚠️  No class names found in {data_yaml}")
        return

    # Build old→new class mapping
    old_to_new: dict[int, int] = {}
    for old_id, name in class_names.items():
        new_id = classify_class_name(name)
        if new_id is not None:
            old_to_new[old_id] = new_id

    # Print mapping
    print(f"    Classes: {class_names}")
    print(f"    Mapping: ", end="")
    for old_id, name in class_names.items():
        new_id = old_to_new.get(old_id)
        if new_id is not None:
            print(f"{name}→{UNIFIED_NAMES[new_id]}", end="  ")
        else:
            print(f"{name}→SKIP", end="  ")
    print()

    if not old_to_new:
        print(f"    ⚠️  No mappable classes found! Skipping.")
        return

    # Process each split
    for split_name, out_split in [("train", "train"), ("val", "valid"),
                                   ("valid", "valid"), ("test", "test")]:
        result = resolve_split_dir(data_yaml, cfg, split_name)
        if result is None:
            continue

        img_dir, lbl_dir = result
        out_img_dir = output_dir / out_split / "images"
        out_lbl_dir = output_dir / out_split / "labels"
        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_lbl_dir.mkdir(parents=True, exist_ok=True)

        img_files = [
            p for p in sorted(img_dir.glob("*.*"))
            if p.suffix in IMAGE_EXTENSIONS
        ]

        copied = 0
        for img_path in img_files:
            # Add prefix to avoid name collisions
            new_stem = f"{dataset_prefix}_{img_path.stem}"
            new_img_name = f"{new_stem}{img_path.suffix}"
            new_lbl_name = f"{new_stem}.txt"

            # Copy image
            dst_img = out_img_dir / new_img_name
            shutil.copy2(img_path, dst_img)

            # Remap and copy label
            src_lbl = lbl_dir / f"{img_path.stem}.txt"
            dst_lbl = out_lbl_dir / new_lbl_name
            remap_label_file(src_lbl, dst_lbl, old_to_new)
            copied += 1

        stats[out_split] = stats.get(out_split, 0) + copied
        if copied > 0:
            print(f"    {split_name:>5}: {copied} images merged → {out_split}")


def create_unified_data_yaml(output_dir: Path) -> Path:
    """Create the unified data.yaml."""
    yaml_path = output_dir / "data.yaml"
    data = {
        "path": str(output_dir.resolve()),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": 2,
        "names": UNIFIED_NAMES,
    }
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    return yaml_path


def main():
    parser = argparse.ArgumentParser(
        description="Merge multiple YOLO datasets into unified bottle+cap dataset"
    )
    parser.add_argument("--input", required=True,
                        help="Root folder containing multiple dataset subfolders")
    parser.add_argument("--output", default="dataset/bottle_cap_merged",
                        help="Output directory for merged dataset")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for shuffling")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    assert input_dir.exists(), f"Input not found: {input_dir}"

    print(f"\n{'='*60}")
    print("  Dataset Merger — Unify to bottle + cap")
    print(f"{'='*60}")
    print(f"  Input  : {input_dir}")
    print(f"  Output : {output_dir}")
    print()

    # Clean output if exists
    if output_dir.exists():
        print(f"  Cleaning existing output: {output_dir}")
        shutil.rmtree(output_dir)

    # Find all datasets
    data_yamls = find_datasets(input_dir)
    if not data_yamls:
        print(f"  ❌ No data.yaml found under {input_dir}")
        print(f"     Make sure each dataset subfolder has a data.yaml")
        return

    print(f"  Found {len(data_yamls)} dataset(s):\n")

    stats: dict[str, int] = {}

    for i, yaml_path in enumerate(data_yamls):
        dataset_name = yaml_path.parent.name
        prefix = f"ds{i}_{dataset_name[:20]}"
        print(f"  [{i+1}/{len(data_yamls)}] {yaml_path.parent}")
        merge_one_dataset(yaml_path, output_dir, prefix, stats)
        print()

    # Create unified data.yaml
    yaml_path = create_unified_data_yaml(output_dir)

    # Final summary
    print(f"  {'='*50}")
    print(f"  ✅ MERGED DATASET READY")
    print(f"  {'='*50}")
    print(f"  data.yaml : {yaml_path}")
    print(f"  Classes   : 0=bottle, 1=cap")
    for split, count in sorted(stats.items()):
        print(f"  {split:>10} : {count} images")
    total = sum(stats.values())
    print(f"  {'total':>10} : {total} images")

    print(f"\n  → NEXT STEP: Run this command to train:")
    print(f"    python training/train_detector.py \\")
    print(f"      --data {yaml_path} \\")
    print(f"      --model yolov8s \\")
    print(f"      --epochs 150 \\")
    print(f"      --profile a5000-balanced \\")
    print(f"      --device 0 \\")
    print(f"      --run-name bottle_cap_det_v2 \\")
    print(f"      --export")
    print()


if __name__ == "__main__":
    main()
