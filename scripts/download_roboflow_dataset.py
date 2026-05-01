"""
scripts/download_roboflow_dataset.py
=====================================
Download and merge Roboflow datasets for bottle+cap detection.

This script downloads multiple Roboflow datasets and merges them
into a single unified dataset with consistent class mapping:
    0: bottle
    1: cap

Usage:
    python scripts/download_roboflow_dataset.py \
        --api-key YOUR_ROBOFLOW_API_KEY \
        --output dataset/bottle_cap_det

    # Or download one specific project:
    python scripts/download_roboflow_dataset.py \
        --api-key YOUR_ROBOFLOW_API_KEY \
        --workspace your-workspace \
        --project your-project \
        --version 1 \
        --output dataset/bottle_cap_det

How to get API key:
    1. Go to https://app.roboflow.com/settings/api
    2. Copy your Private API Key

How to find datasets:
    1. Go to https://universe.roboflow.com
    2. Search: "bottle cap detection"
    3. Pick datasets with "bottle" and "cap" classes
    4. Click the dataset → look at the URL for workspace/project info
    5. Or click "Download Dataset" → "Show download code" → copy the snippet
"""

from __future__ import annotations

import argparse
import shutil
import yaml
from pathlib import Path


def download_single_dataset(
    api_key: str,
    workspace: str,
    project: str,
    version: int,
    output_dir: Path,
    format: str = "yolov8",
):
    """Download a single Roboflow dataset."""
    from roboflow import Roboflow

    rf = Roboflow(api_key=api_key)
    proj = rf.workspace(workspace).project(project)
    ver = proj.version(version)

    print(f"\n  Downloading: {workspace}/{project} v{version}")
    dataset = ver.download(format, location=str(output_dir))
    print(f"  → Saved to: {output_dir}")
    return dataset


def remap_labels(labels_dir: Path, class_mapping: dict[int, int]):
    """Remap class IDs in YOLO label files."""
    if not labels_dir.exists():
        return

    for lbl_path in sorted(labels_dir.glob("*.txt")):
        lines = lbl_path.read_text(encoding="utf-8").strip().splitlines()
        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            old_cls = int(float(parts[0]))
            if old_cls in class_mapping:
                parts[0] = str(class_mapping[old_cls])
                new_lines.append(" ".join(parts))
            # Skip classes not in mapping (we only want bottle + cap)
        lbl_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")


def merge_datasets(source_dir: Path, target_dir: Path, prefix: str = ""):
    """Merge one dataset into the target, adding a prefix to avoid name collisions."""
    for split in ["train", "valid", "val", "test"]:
        # Normalize "valid" → "val" in target
        target_split = "valid" if split == "val" else split

        src_images = source_dir / split / "images"
        src_labels = source_dir / split / "labels"

        if not src_images.exists():
            continue

        tgt_images = target_dir / target_split / "images"
        tgt_labels = target_dir / target_split / "labels"
        tgt_images.mkdir(parents=True, exist_ok=True)
        tgt_labels.mkdir(parents=True, exist_ok=True)

        for img_path in src_images.glob("*.*"):
            if img_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                new_name = f"{prefix}{img_path.name}" if prefix else img_path.name
                shutil.copy2(img_path, tgt_images / new_name)

                lbl_path = src_labels / f"{img_path.stem}.txt"
                if lbl_path.exists():
                    new_lbl_name = f"{prefix}{img_path.stem}.txt"
                    shutil.copy2(lbl_path, tgt_labels / new_lbl_name)


def create_data_yaml(output_dir: Path):
    """Create a unified data.yaml for the merged dataset."""
    data = {
        "path": str(output_dir.resolve()),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": 2,
        "names": {0: "bottle", 1: "cap"},
    }

    yaml_path = output_dir / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    print(f"\n  Created: {yaml_path}")
    return yaml_path


def count_dataset(output_dir: Path):
    """Print dataset statistics."""
    print(f"\n  ── Dataset Summary ──")
    for split in ["train", "valid", "test"]:
        imgs_dir = output_dir / split / "images"
        if imgs_dir.exists():
            n_imgs = len(list(imgs_dir.glob("*.*")))
            print(f"  {split:>5}: {n_imgs} images")
        else:
            print(f"  {split:>5}: (not found)")


def main():
    parser = argparse.ArgumentParser(
        description="Download Roboflow datasets for bottle+cap detection"
    )
    parser.add_argument("--api-key", required=True,
                        help="Your Roboflow API key")
    parser.add_argument("--workspace", default="",
                        help="Roboflow workspace ID")
    parser.add_argument("--project", default="",
                        help="Roboflow project ID")
    parser.add_argument("--version", type=int, default=1,
                        help="Dataset version number")
    parser.add_argument("--output", default="dataset/bottle_cap_det",
                        help="Output directory for merged dataset")
    args = parser.parse_args()

    output_dir = Path(args.output)

    print(f"\n{'='*55}")
    print("  Roboflow Dataset Downloader")
    print(f"{'='*55}")

    if args.workspace and args.project:
        # Download specific dataset
        download_single_dataset(
            args.api_key, args.workspace, args.project,
            args.version, output_dir,
        )
    else:
        print("\n  No --workspace/--project specified.")
        print("  Go to https://universe.roboflow.com and search 'bottle cap detection'")
        print("  Then run with: --workspace <id> --project <id> --version <num>")
        print("\n  Example:")
        print("    python scripts/download_roboflow_dataset.py \\")
        print("      --api-key YOUR_KEY \\")
        print("      --workspace my-workspace \\")
        print("      --project bottle-cap-detection \\")
        print("      --version 1 \\")
        print("      --output dataset/bottle_cap_det")
        return

    create_data_yaml(output_dir)
    count_dataset(output_dir)

    print(f"\n  ✅ Dataset ready at: {output_dir.resolve()}")
    print(f"  → Next: python training/train_detector.py --data {output_dir}/data.yaml\n")


if __name__ == "__main__":
    main()
