"""
Script 01: Merge class-sorted folders into flat YOLO training structure.
Run this FIRST before any training.

Input Structure:
    <split>/<class_name>/images/<image_file>
    <split>/<class_name>/labels/<label_file>

Output Structure (YOLO-ready):
    <split>/images/<image_file>
    <split>/labels/<label_file>
"""

import os
import shutil
from pathlib import Path
import argparse


DATASET_DIR = r"D:\Yolo Dataset\bottle_cap_sdp.v7i.yolov11"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ["train", "valid", "test"]
SKIP_DIRS = {"images", "labels", "lables", "Labels", "Images"}


def find_subdir(parent, *candidates):
    """Return the first existing subdirectory from a list of candidate names."""
    for name in candidates:
        path = parent / name
        if path.exists():
            return path
    return None


def merge_dataset(dataset_dir, move_files=False):
    dataset_path = Path(dataset_dir).resolve()
    if not dataset_path.exists():
        print(f"[Error] Dataset directory not found: {dataset_path}")
        return

    action = "Moving" if move_files else "Copying"
    print("=" * 70)
    print(f"  MERGE TO YOLO FORMAT  ({action} mode)")
    print(f"  Dataset: {dataset_path}")
    print("=" * 70)

    total_imgs, total_lbls = 0, 0

    for split in SPLITS:
        split_dir = dataset_path / split
        if not split_dir.exists():
            print(f"\n[Skip] '{split}' directory not found.")
            continue

        dest_imgs = split_dir / "images"
        dest_lbls = split_dir / "labels"
        dest_imgs.mkdir(parents=True, exist_ok=True)
        dest_lbls.mkdir(parents=True, exist_ok=True)

        class_dirs = [d for d in split_dir.iterdir()
                      if d.is_dir() and d.name not in SKIP_DIRS]

        print(f"\n[{split.upper()}]  {len(class_dirs)} class folders found")

        split_imgs, split_lbls = 0, 0

        for class_dir in sorted(class_dirs, key=lambda d: d.name):
            img_dir = find_subdir(class_dir, "images", "Images")
            lbl_dir = find_subdir(class_dir, "labels", "lables", "Labels")

            if img_dir is None:
                continue

            images = [p for p in img_dir.iterdir()
                      if p.suffix.lower() in IMAGE_EXTENSIONS]

            for img in images:
                dst_img = dest_imgs / img.name
                if move_files:
                    shutil.move(str(img), str(dst_img))
                else:
                    shutil.copy2(str(img), str(dst_img))
                split_imgs += 1

                if lbl_dir:
                    lbl = lbl_dir / f"{img.stem}.txt"
                    if lbl.exists():
                        dst_lbl = dest_lbls / lbl.name
                        if move_files:
                            shutil.move(str(lbl), str(dst_lbl))
                        else:
                            shutil.copy2(str(lbl), str(dst_lbl))
                        split_lbls += 1

            print(f"  '{class_dir.name}': {len(images)} items merged")

        print(f"  -> {split_imgs} images | {split_lbls} labels in {split}/images & {split}/labels")
        total_imgs += split_imgs
        total_lbls += split_lbls

    print("\n" + "=" * 70)
    print(f"  MERGE COMPLETE")
    print(f"  Total Images : {total_imgs}")
    print(f"  Total Labels : {total_lbls}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", default=DATASET_DIR)
    parser.add_argument("--move", action="store_true", help="Move instead of copy")
    args = parser.parse_args()
    merge_dataset(args.dataset, move_files=args.move)
