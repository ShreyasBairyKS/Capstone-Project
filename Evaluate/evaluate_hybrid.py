"""
Evaluate/evaluate_hybrid.py
============================
End-to-end evaluation of the hybrid detection + classification pipeline.

Evaluates:
    1. Detection metrics (mAP) via YOLOv8 val
    2. Classification metrics (Precision, Recall, F1) on cap crops
    3. End-to-end verdict accuracy
    4. Latency benchmark

Usage:
    python Evaluate/evaluate_hybrid.py \
        --det-weights runs/detect/bottle_cap_det_v2/weights/best.pt \
        --cls-weights models/cap_classifier_best.pth \
        --det-data configs/detection_data.yaml \
        --cls-data data/caps \
        --device 0
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def evaluate_detection(weights: str, data_yaml: str, conf: float, iou: float):
    """Run YOLOv8 validation on detection test split."""
    from ultralytics import YOLO

    print("\n[1/3] Evaluating detection model...")
    model = YOLO(weights)
    metrics = model.val(
        data=data_yaml, split="test", conf=conf, iou=iou,
        save=True, plots=True,
        project="runs/detect", name="hybrid_det_eval",
        verbose=True,
    )

    box = getattr(metrics, "box", None)
    summary = {}
    if box:
        for key, attr in {"precision": "mp", "recall": "mr",
                          "mAP50": "map50", "mAP50_95": "map"}.items():
            val = getattr(box, attr, None)
            if val is not None:
                summary[key] = float(val)

    print(f"\n  Detection Results:")
    for k, v in summary.items():
        print(f"    {k}: {v:.4f}")

    return summary


def evaluate_classifier(cls_weights: str, data_root: str, device: str, imgsz: int):
    """Evaluate classifier on val/test set."""
    print("\n[2/3] Evaluating classifier...")

    # Import from training script
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from training.train_cap_classifier import (
        CapDataset, build_classifier, CLS_NAMES, FocalLoss,
    )

    torch_device = torch.device(f"cuda:{device}" if device.isdigit() else device)
    data_path = Path(data_root)

    # Try test split first, fall back to val
    test_split = "test" if (data_path / "test").exists() else "val"
    test_ds = CapDataset(data_path, test_split, imgsz)

    if len(test_ds) == 0:
        print(f"  No test samples found in {data_path / test_split}")
        return {}

    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=4)

    # Load model
    checkpoint = torch.load(cls_weights, map_location=torch_device, weights_only=False)
    model = build_classifier(num_classes=len(CLS_NAMES))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(torch_device)
    model.eval()

    tp = fp = fn = tn = 0  # for defective class (index 1)
    correct = 0
    total = 0

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(torch_device), labels.to(torch_device)
            preds = model(imgs).argmax(1)
            correct += (preds == labels).sum().item()
            total += imgs.size(0)

            for p, g in zip(preds.cpu().numpy(), labels.cpu().numpy()):
                if g == 1 and p == 1: tp += 1
                elif g == 0 and p == 1: fp += 1
                elif g == 1 and p == 0: fn += 1
                else: tn += 1

    acc = correct / max(total, 1)
    recall = tp / max(tp + fn, 1)
    precision = tp / max(tp + fp, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)

    print(f"\n  Classifier Results ({test_split} set, {total} samples):")
    print(f"  ┌──────────────────────────────────────┐")
    print(f"  │  Accuracy   : {acc:.4f}               │")
    print(f"  │  Precision  : {precision:.4f}          │")
    print(f"  │  Recall     : {recall:.4f} ← critical │")
    print(f"  │  F1 Score   : {f1:.4f}                │")
    print(f"  │  TP={tp}  FP={fp}  FN={fn}  TN={tn}   │")
    print(f"  └──────────────────────────────────────┘")

    if fn > 0:
        print(f"\n  ⚠️  {fn} defective caps MISSED (False Negatives)")
        print(f"     → Lower detection conf or add more defective training data")

    return {
        "accuracy": acc, "precision": precision, "recall": recall, "f1": f1,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn, "total": total,
    }


def benchmark_hybrid(det_weights: str, cls_weights: str,
                     test_images: str, device: str, n_samples: int = 30):
    """Benchmark end-to-end hybrid pipeline latency."""
    print(f"\n[3/3] Benchmarking hybrid pipeline speed ({n_samples} images)...")

    from inference.hybrid_inference import HybridInspector

    inspector = HybridInspector(
        det_weights=det_weights,
        cls_weights=cls_weights,
        device=device,
    )

    images_dir = Path(test_images)
    imgs = sorted(p for p in images_dir.glob("*.*") if p.suffix.lower() in IMAGE_EXTENSIONS)
    imgs = imgs[:n_samples]

    if not imgs:
        print(f"  No images found for benchmark at {images_dir}")
        return {}

    import cv2

    # Warm-up
    warm_img = cv2.imread(str(imgs[0]))
    if warm_img is not None:
        inspector.inspect(warm_img)

    latencies = []
    for img_path in imgs:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        t0 = time.perf_counter()
        inspector.inspect(img)
        latencies.append((time.perf_counter() - t0) * 1000)

    if not latencies:
        return {}

    avg_ms = float(np.mean(latencies))
    p95_ms = float(np.percentile(latencies, 95))
    fps = 1000.0 / avg_ms

    print(f"\n  Hybrid Pipeline Speed:")
    print(f"    Average latency : {avg_ms:.1f} ms")
    print(f"    P95 latency     : {p95_ms:.1f} ms")
    print(f"    Throughput      : {fps:.1f} FPS")

    if fps >= 30:
        print(f"    ✅ Real-time ready (≥30 FPS)")
    elif fps >= 10:
        print(f"    ⚡ Near-real-time (10-30 FPS)")
    else:
        print(f"    ⚠️  Below real-time — consider lighter models")

    return {"avg_ms": avg_ms, "p95_ms": p95_ms, "fps": fps}


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate hybrid detection + classification pipeline"
    )
    parser.add_argument("--det-weights", required=True,
                        help="Path to detector .pt weights")
    parser.add_argument("--cls-weights", required=True,
                        help="Path to classifier .pth weights")
    parser.add_argument("--det-data", default="configs/detection_data.yaml",
                        help="Detection data.yaml path")
    parser.add_argument("--cls-data", default="data/caps",
                        help="Classification dataset root")
    parser.add_argument("--test-images", default="",
                        help="Images folder for speed benchmark")
    parser.add_argument("--det-conf", type=float, default=0.25)
    parser.add_argument("--det-iou", type=float, default=0.45)
    parser.add_argument("--cls-imgsz", type=int, default=224)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print("  Hybrid Pipeline — Full Evaluation")
    print(f"{'='*60}")

    report = {}

    # 1. Detection eval
    det_data = Path(args.det_data)
    if det_data.exists():
        report["detection"] = evaluate_detection(
            args.det_weights, str(det_data), args.det_conf, args.det_iou
        )
    else:
        print(f"\n  [SKIP] Detection data.yaml not found: {det_data}")

    # 2. Classifier eval
    cls_data = Path(args.cls_data)
    if cls_data.exists():
        report["classification"] = evaluate_classifier(
            args.cls_weights, str(cls_data), args.device, args.cls_imgsz
        )

    # 3. Speed benchmark
    test_imgs = args.test_images
    if not test_imgs and det_data.exists():
        # Try to find test images from det data
        import yaml
        with open(det_data) as f:
            cfg = yaml.safe_load(f) or {}
        test_path = cfg.get("test", "")
        if test_path:
            candidate = (det_data.parent / test_path).resolve()
            if candidate.exists():
                test_imgs = str(candidate)

    if test_imgs and Path(test_imgs).exists():
        report["speed"] = benchmark_hybrid(
            args.det_weights, args.cls_weights, test_imgs, args.device
        )

    # Save report
    output_dir = Path("runs/detect/hybrid_eval")
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "hybrid_eval_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\n✅ Hybrid evaluation complete.")
    print(f"   Report → {report_path}\n")


if __name__ == "__main__":
    main()
