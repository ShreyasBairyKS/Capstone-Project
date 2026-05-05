"""
scripts/merge_water_datasets.py
================================
Merge waterfill2 + waterfill3 into a unified water surface detection dataset.

Rules enforced by this script:
    1. Only dataset/Beverages/waterfill2 and dataset/Beverages/waterfill3 are used.
    2. waterfill2 is copied in full.
    3. waterfill3 is sampled by split with approximate proportions:
           train  -> 30%
           test   -> 20%
           valid  -> 10%
    4. The merged dataset is written to Beverages/waterfill_merge.
    5. Output follows train/test/valid formatting.

Both datasets detect the water surface line:
    waterfill2: 'Surface-of-water-in-bottle' -> class 0: water_surface
    waterfill3: 'Level'                      -> class 0: water_surface

Usage:
    python scripts/merge_water_datasets.py `
        --ds1 "E:\P-25 Vision Food ai\dataset\Beverages\waterfill2" `
        --ds2 "E:\P-25 Vision Food ai\dataset\Beverages\waterfill3" `
        --output "E:\P-25 Vision Food ai\dataset\Beverages\waterfill_merge"

    # Then train:
    yolo train model=yolov8n.pt `
        data="E:\P-25 Vision Food ai\dataset\Beverages\waterfill_merge\data.yaml" `
        epochs=110 imgsz=640 device=0 `
        project=runs/detect name=water_surface_v1
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

import yaml

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".PNG", ".JPEG"}
SPLITS = ["train", "test", "valid"]
DS2_RATIOS = {
    "train": 0.30,
    "test": 0.20,
    "valid": 0.10,
}


def find_images_and_labels(dataset_root: Path):
    """Find image/label pairs across train/test/valid splits."""
    pairs = {}
    for split in SPLITS:
        img_dir = dataset_root / split / "images"
        lbl_dir = dataset_root / split / "labels"
        if not img_dir.exists():
            # Try alternate path: images directly under split
            img_dir = dataset_root / split
            lbl_dir = dataset_root / split

        split_pairs = []
        if img_dir.exists():
            for img_path in sorted(img_dir.glob("*.*")):
                if img_path.suffix.lower() in IMAGE_EXTENSIONS:
                    lbl_path = lbl_dir / f"{img_path.stem}.txt"
                    if lbl_path.exists():
                        split_pairs.append((img_path, lbl_path))
        pairs[split] = split_pairs
    return pairs


def copy_with_remap(pairs: list[tuple[Path, Path]],
                    out_img_dir: Path,
                    out_lbl_dir: Path,
                    src_class_id: int,
                    dst_class_id: int,
                    prefix: str,
                    sample_ratio: float = 1.0,
                    seed: int = 42):
    """
    Copy image+label pairs, remapping src_class_id → dst_class_id.
    Optionally sample a fraction of the pairs.
    """
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    if sample_ratio < 1.0:
        if not pairs:
            return 0
        rng = random.Random(seed)
        n = min(len(pairs), max(1, int(len(pairs) * sample_ratio)))
        pairs = rng.sample(pairs, n)

    count = 0
    for img_path, lbl_path in pairs:
        # Unique filename with dataset prefix to avoid collisions
        stem = f"{prefix}_{img_path.stem}"
        suffix = img_path.suffix.lower() or ".jpg"

        # Copy image
        shutil.copy2(img_path, out_img_dir / f"{stem}{suffix}")

        # Remap labels
        lines = lbl_path.read_text(encoding="utf-8").strip().splitlines()
        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue
            cls_id = int(float(parts[0]))
            if cls_id == src_class_id:
                parts[0] = str(dst_class_id)
            new_lines.append(" ".join(parts))

        out_lbl = out_lbl_dir / f"{stem}.txt"
        out_lbl.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
        count += 1

    return count


def main():
    parser = argparse.ArgumentParser(
        description="Merge water surface detection datasets"
    )
    parser.add_argument("--ds1", required=True,
                        help="Path to waterfill2 dataset root (used 100%%)")
    parser.add_argument("--ds2", required=True,
                        help="Path to waterfill3 dataset root")
    parser.add_argument("--output", default="dataset/Beverages/waterfill_merge",
                        help="Output merged dataset path (default: dataset/Beverages/waterfill_merge)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for sampling")
    args = parser.parse_args()

    ds1 = Path(args.ds1)
    ds2 = Path(args.ds2)
    out = Path(args.output)

    assert ds1.exists(), f"Dataset 1 not found: {ds1}"
    assert ds2.exists(), f"Dataset 2 not found: {ds2}"

    print(f"\n{'='*55}")
    print("  Water Surface Dataset Merger")
    print(f"{'='*55}")
    print(f"  DS1 (100%)     : {ds1.name}")
    print(f"  DS2 sampled    : {ds2.name}")
    print(f"  Output         : {out}")
    print(f"  Target class   : 0 → water_surface")
    print()

    ds1_pairs = find_images_and_labels(ds1)
    ds2_pairs = find_images_and_labels(ds2)

    total_counts = {}
    for split in SPLITS:
        out_img = out / split / "images"
        out_lbl = out / split / "labels"
        ds2_ratio = DS2_RATIOS[split]

        n1 = copy_with_remap(
            ds1_pairs.get(split, []),
            out_img, out_lbl,
            src_class_id=0, dst_class_id=0,
            prefix="wf2",
            sample_ratio=1.0,
            seed=args.seed,
        )

        n2 = copy_with_remap(
            ds2_pairs.get(split, []),
            out_img, out_lbl,
            src_class_id=0, dst_class_id=0,
            prefix="wf3",
            sample_ratio=ds2_ratio,
            seed=args.seed,
        )

        total = n1 + n2
        total_counts[split] = total
        print(f"  {split:<6}: {n1} from ds1 + {n2} from ds2 ({int(ds2_ratio*100)}%) = {total} total")

    # Write data.yaml
    data_yaml = {
        "train": f"../train/images",
        "val": f"../valid/images",
        "valid": f"../valid/images",
        "test": f"../test/images",
        "nc": 1,
        "names": ["water_surface"],
    }
    yaml_path = out / "data.yaml"
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with open(yaml_path, "w") as f:
        yaml.dump(data_yaml, f, default_flow_style=False, sort_keys=False)

    grand_total = sum(total_counts.values())
    print(f"\n  ── Summary ──")
    print(f"  Grand total  : {grand_total} image-label pairs")
    print(f"  data.yaml    : {yaml_path}")
    print(f"\n  → NEXT: Train water surface detector:")
    print(f"    yolo train model=yolov8n.pt data={yaml_path} epochs=100 imgsz=640 device=0 project=runs/detect name=water_surface_v1")
    print()


if __name__ == "__main__":
    main()
