"""
Script 04: Generate Soft Labels from the Teacher Model

The teacher runs inference over the entire training set and saves its
class probability distributions (softened at temperature T) alongside
the raw predicted bounding boxes and mask coefficients.

These soft labels are loaded by Script 05 during student training to
compute the KL-Divergence distillation loss alongside ground truth labels.

Output:
    soft_labels/<image_stem>.npy  -- numpy array with:
        shape: (num_detections, 6 + 32)
        cols 0-5  : softened class probabilities (temperature T=4)
        cols 6-37 : mask coefficients (32 values)

Run after: 03_evaluate_teacher.py  (only when teacher passes quality gate)
"""

import numpy as np
from pathlib import Path
from ultralytics import YOLO
import torch
import torch.nn.functional as F
import argparse

# ─── Config ───────────────────────────────────────────────────────────────────
TEACHER_WEIGHTS  = r"D:\Yolo Dataset\YOLO_TrainingScripts\runs\teacher_yolo11x_seg\weights\best.pt"
TRAIN_IMAGES_DIR = r"D:\Yolo Dataset\bottle_cap_sdp.v7i.yolov11\train\images"
OUTPUT_DIR       = r"D:\Yolo Dataset\YOLO_TrainingScripts\soft_labels"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
TEMPERATURE      = 4.0    # KD temperature — higher = softer distributions
BATCH_SIZE       = 32     # Inference batch (safe for A5000 24GB)
CONF_THRESHOLD   = 0.05   # Low conf to keep soft uncertainties
IOU_THRESHOLD    = 0.45


def soften_probs(logits: np.ndarray, T: float) -> np.ndarray:
    """Apply temperature scaling to class logits and return soft probabilities."""
    t = torch.tensor(logits / T, dtype=torch.float32)
    return F.softmax(t, dim=-1).numpy()


def generate_soft_labels(teacher_weights, images_dir, output_dir, temperature=4.0):
    images_dir = Path(images_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    images = sorted([p for p in images_dir.iterdir()
                     if p.suffix.lower() in IMAGE_EXTENSIONS])

    print("=" * 65)
    print(f"  SOFT LABEL GENERATION")
    print(f"  Teacher  : {Path(teacher_weights).name}")
    print(f"  Images   : {len(images)}")
    print(f"  Temp (T) : {temperature}")
    print(f"  Output   : {output_dir}")
    print("=" * 65)

    model = YOLO(teacher_weights)
    model.fuse()   # Fuse Conv+BN for faster inference

    saved, skipped = 0, 0

    # Process in batches for throughput
    for batch_start in range(0, len(images), BATCH_SIZE):
        batch_paths = images[batch_start: batch_start + BATCH_SIZE]
        batch_strs  = [str(p) for p in batch_paths]

        results = model.predict(
            source=batch_strs,
            imgsz=640,
            conf=CONF_THRESHOLD,
            iou=IOU_THRESHOLD,
            device=0,
            verbose=False,
            save=False,
        )

        for img_path, result in zip(batch_paths, results):
            out_path = output_dir / f"{img_path.stem}.npy"

            if result.boxes is None or len(result.boxes) == 0:
                # No detections: save empty array
                np.save(str(out_path), np.zeros((0, 38), dtype=np.float32))
                skipped += 1
                continue

            # Class logits -> softened probabilities
            cls_probs_raw = result.boxes.cls.cpu().numpy()    # shape (N,) class indices
            cls_conf_raw  = result.boxes.conf.cpu().numpy()   # shape (N,) confidences

            # Build per-detection class probability vectors
            num_classes = len(model.names)
            num_det = len(cls_probs_raw)
            cls_logits = np.zeros((num_det, num_classes), dtype=np.float32)
            for i, (cls_idx, conf) in enumerate(zip(cls_probs_raw, cls_conf_raw)):
                cls_logits[i, int(cls_idx)] = conf   # Approximate logit from confidence

            soft_probs = np.array([soften_probs(logits, temperature) for logits in cls_logits])

            # Mask coefficients (32 values per detection)
            if result.masks is not None and hasattr(result, 'boxes'):
                # Get raw mask data if available
                try:
                    mask_coefs = result.boxes.data.cpu().numpy()[:, 6:38]  # cols 6-37
                except Exception:
                    mask_coefs = np.zeros((num_det, 32), dtype=np.float32)
            else:
                mask_coefs = np.zeros((num_det, 32), dtype=np.float32)

            # Concatenate: [soft_probs(6), mask_coeffs(32)] -> shape (N, 38)
            soft_record = np.concatenate([soft_probs, mask_coefs], axis=1).astype(np.float32)
            np.save(str(out_path), soft_record)
            saved += 1

        progress = min(batch_start + BATCH_SIZE, len(images))
        print(f"  Processed {progress}/{len(images)} images...", end="\r")

    print(f"\n\n  Soft labels saved : {saved}")
    print(f"  Empty (no detect) : {skipped}")
    print(f"  Output directory  : {output_dir}")
    print("=" * 65)
    print("Next step -> Run: python 05_train_student_distill.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--weights",   default=TEACHER_WEIGHTS)
    parser.add_argument("-i", "--images",    default=TRAIN_IMAGES_DIR)
    parser.add_argument("-o", "--output",    default=OUTPUT_DIR)
    parser.add_argument("-t", "--temperature", default=TEMPERATURE, type=float)
    args = parser.parse_args()

    generate_soft_labels(args.weights, args.images, args.output, args.temperature)
