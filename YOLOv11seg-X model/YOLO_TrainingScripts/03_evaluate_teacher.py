"""
Script 03: Evaluate the Trained Teacher Model
Validates on the test split and prints:
  - Overall Box mAP50 / mAP50-95
  - Overall Mask mAP50 / mAP50-95
  - Per-class Box and Mask mAP50

Run after: 02_train_teacher.py
"""

from ultralytics import YOLO
from pathlib import Path
import argparse

# ─── Paths ────────────────────────────────────────────────────────────────────
DEFAULT_WEIGHTS = r"E:\Bottle_cap defect detection team 25\YOLOv11seg-X model\YOLO_TrainingScripts\runs\teacher_yolo11x_seg\weights\best.pt"
DATASET_YAML    = r"E:\Bottle_cap defect detection team 25\YOLOv11seg-X model\bottle_cap_sdp.v7i.yolov11\data.yaml"
CLASS_NAMES     = ["damaged_cap", "good_cap", "misplaced_cap", "no_cap", "open_cap", "wet_cap"]


def evaluate_teacher(weights_path, split="test"):
    weights_path = Path(weights_path)
    if not weights_path.exists():
        print(f"[Error] Weights not found: {weights_path}")
        print("Run 02_train_teacher.py first.")
        return

    print("=" * 65)
    print(f"  TEACHER MODEL EVALUATION")
    print(f"  Weights : {weights_path.name}")
    print(f"  Split   : {split}")
    print("=" * 65)

    model = YOLO(str(weights_path))
    metrics = model.val(
        data=DATASET_YAML,
        split=split,
        imgsz=640,
        batch=16,
        device=0,
        verbose=True,
        plots=True,
        save_json=True,
    )

    # ── Overall Metrics ───────────────────────────────────────────────────────
    print("\n" + "─" * 65)
    print("  OVERALL RESULTS")
    print("─" * 65)
    print(f"  Box  mAP50     : {metrics.box.map50:.4f}")
    print(f"  Box  mAP50-95  : {metrics.box.map:.4f}")
    print(f"  Mask mAP50     : {metrics.seg.map50:.4f}")
    print(f"  Mask mAP50-95  : {metrics.seg.map:.4f}")

    # ── Per-Class Metrics ─────────────────────────────────────────────────────
    print("\n" + "─" * 65)
    print(f"  {'Class':<20} {'Box mAP50':>12} {'Mask mAP50':>12}")
    print("─" * 65)

    for i, name in enumerate(CLASS_NAMES):
        try:
            box_map  = metrics.box.ap50[i]
            mask_map = metrics.seg.ap50[i]
            status = ""
            if mask_map < 0.80:
                status = " << BELOW TARGET"
            print(f"  {name:<20} {box_map:>12.4f} {mask_map:>12.4f}{status}")
        except (IndexError, AttributeError):
            print(f"  {name:<20} {'N/A':>12} {'N/A':>12}")

    # ── Evaluation Gate ───────────────────────────────────────────────────────
    print("\n" + "─" * 65)
    box_ok  = metrics.box.map50 >= 0.90
    mask_ok = metrics.seg.map50 >= 0.88

    if box_ok and mask_ok:
        print("  [PASS] Teacher meets quality gate.")
        print("  Next step -> Run: python 04_generate_soft_labels.py")
    else:
        print("  [WARN] Teacher below quality threshold.")
        if not box_ok:
            print(f"    Box  mAP50 = {metrics.box.map50:.4f} (target >= 0.90)")
        if not mask_ok:
            print(f"    Mask mAP50 = {metrics.seg.map50:.4f} (target >= 0.88)")
        print("  Consider training more epochs or adjusting hyperparameters.")

    print("=" * 65)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--weights", default=DEFAULT_WEIGHTS, help="Path to teacher best.pt")
    parser.add_argument("-s", "--split", default="test", choices=["train", "val", "test"])
    args = parser.parse_args()
    evaluate_teacher(args.weights, args.split)
