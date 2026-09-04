"""
Script 06: Evaluate and Compare Teacher vs Student Models

Runs validation on the test split for both models and prints a
side-by-side comparison table including:
  - Overall Box / Mask mAP50 and mAP50-95
  - Per-class Mask mAP50
  - Model size (MB) and parameter count
  - Knowledge retention percentage

Run after: 05_train_student_distill.py
"""

from ultralytics import YOLO
from pathlib import Path
import argparse
import os

# ─── Paths ────────────────────────────────────────────────────────────────────
TEACHER_WEIGHTS = r"D:\Yolo Dataset\YOLO_TrainingScripts\runs\teacher_yolo11x_seg\weights\best.pt"
STUDENT_WEIGHTS = r"D:\Yolo Dataset\YOLO_TrainingScripts\runs\student_yolo11n_distilled\weights\best.pt"
DATASET_YAML    = r"D:\Yolo Dataset\bottle_cap_sdp.v7i.yolov11\data.yaml"
CLASS_NAMES     = ["damaged_cap", "good_cap", "misplaced_cap", "no_cap", "open_cap", "wet_cap"]


def get_model_size_mb(weights_path):
    return os.path.getsize(weights_path) / (1024 * 1024)


def count_params(model):
    return sum(p.numel() for p in model.model.parameters())


def run_eval(weights_path, split="test"):
    model = YOLO(weights_path)
    metrics = model.val(
        data=DATASET_YAML,
        split=split,
        imgsz=640,
        batch=16,
        device=0,
        verbose=False,
        plots=False,
    )
    return model, metrics


def compare(teacher_weights, student_weights, split="test"):
    print("=" * 75)
    print("  TEACHER vs STUDENT COMPARISON")
    print("=" * 75)

    print(f"\n[Evaluating Teacher]  {Path(teacher_weights).name}")
    teacher_model, t_metrics = run_eval(teacher_weights, split)

    print(f"\n[Evaluating Student]  {Path(student_weights).name}")
    student_model, s_metrics = run_eval(student_weights, split)

    # ── Model Info ────────────────────────────────────────────────────────────
    t_size   = get_model_size_mb(teacher_weights)
    s_size   = get_model_size_mb(student_weights)
    t_params = count_params(teacher_model)
    s_params = count_params(student_model)

    print("\n" + "─" * 75)
    print(f"  {'Metric':<30} {'Teacher (yolo11x)':>20} {'Student (yolo11n)':>20}")
    print("─" * 75)

    def row(label, t_val, s_val, fmt=".4f", target=None):
        t_str = f"{t_val:{fmt}}"
        s_str = f"{s_val:{fmt}}"
        retention = f" ({s_val / t_val * 100:.1f}%)" if t_val > 0 else ""
        flag = ""
        if target is not None and s_val < target:
            flag = " << BELOW TARGET"
        print(f"  {label:<30} {t_str:>20} {s_str + retention:>20}{flag}")

    row("Box  mAP50",     t_metrics.box.map50, s_metrics.box.map50, target=0.87)
    row("Box  mAP50-95",  t_metrics.box.map,   s_metrics.box.map,   target=0.0)
    row("Mask mAP50",     t_metrics.seg.map50, s_metrics.seg.map50, target=0.85)
    row("Mask mAP50-95",  t_metrics.seg.map,   s_metrics.seg.map,   target=0.0)

    print("─" * 75)
    print(f"  {'Model Size (MB)':<30} {t_size:>20.1f} {s_size:>20.1f}")
    print(f"  {'Parameters (M)':<30} {t_params/1e6:>20.2f} {s_params/1e6:>20.2f}")
    print(f"  {'Size Reduction':<30} {'—':>20} {f'{t_size/s_size:.1f}x smaller':>20}")
    print(f"  {'Param Reduction':<30} {'—':>20} {f'{t_params/s_params:.1f}x fewer':>20}")

    # ── Per-Class Mask mAP50 ──────────────────────────────────────────────────
    print("\n" + "─" * 75)
    print(f"  {'Class':<25} {'Teacher Mask mAP50':>22} {'Student Mask mAP50':>22}")
    print("─" * 75)

    for i, name in enumerate(CLASS_NAMES):
        try:
            t_map = t_metrics.seg.ap50[i]
            s_map = s_metrics.seg.ap50[i]
            retention = f"({s_map / t_map * 100:.1f}%)" if t_map > 0 else ""
            flag = " << LOW" if s_map < 0.80 else ""
            print(f"  {name:<25} {t_map:>22.4f} {s_map:>16.4f} {retention}{flag}")
        except (IndexError, AttributeError):
            print(f"  {name:<25} {'N/A':>22} {'N/A':>22}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "─" * 75)
    overall_retention = s_metrics.seg.map50 / t_metrics.seg.map50 * 100
    print(f"  Overall Mask mAP50 Retention : {overall_retention:.1f}%")
    if overall_retention >= 94:
        print("  [PASS] Excellent distillation — >94% knowledge retained.")
    elif overall_retention >= 90:
        print("  [PASS] Good distillation — >90% knowledge retained.")
    else:
        print("  [WARN] Knowledge loss detected. Consider increasing beta (KD weight) or re-tuning temperature.")

    print("=" * 75)
    print("Next step -> Run: python 07_export_student.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher", default=TEACHER_WEIGHTS)
    parser.add_argument("--student", default=STUDENT_WEIGHTS)
    parser.add_argument("--split",   default="test", choices=["train", "val", "test"])
    args = parser.parse_args()
    compare(args.teacher, args.student, args.split)
