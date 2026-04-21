"""
bottle-cap-evaluate.py
======================
Evaluates the trained YOLOv11 beverage cap-quality model on the test set.

Produces:
    • mAP@0.5 and mAP@0.5:0.95
    • Precision, Recall, F1 per class
    • Confusion matrix
    • Per-image inference results with defect/no-defect classification
    • Summary report saved to runs/detect/bottle_cap_eval/

Run:
    python Evaluate/bottle-cap-evaluate.py --weights runs/detect/bottle_cap_defect/weights/best.pt
                                            --data    dataset/Beverages/bottleDefect.v1-first.yolov11-cap/data.yaml
                                            --conf    0.25
                                            --iou     0.45

Threshold guidance:
    --conf  0.25   → Default. Lower = more detections (more FP, fewer FN).
                    Raise to 0.4–0.5 to reduce false positives in production.
    --iou   0.45   → NMS IoU threshold. Higher = stricter duplicate suppression.
"""

import argparse
import json
from pathlib import Path
import numpy as np
import yaml
from ultralytics import YOLO


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
DEFAULT_DATA_YAML = "dataset/Beverages/bottleDefect.v1-first.yolov11-cap/data.yaml"
DEFAULT_DEFECT_CLASSES = "defectCap,noCap"


# ─────────────────────────────────────────────────────────────────────────────
# METRIC HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def compute_binary_metrics(tp: int, fp: int, fn: int) -> dict:
    """
    Returns precision, recall, F1 for binary defect/no-defect classification.
    These are computed at the image level (not bounding-box level).
    """
    precision = tp / (tp + fp + 1e-9)
    recall    = tp / (tp + fn + 1e-9)
    f1        = 2 * precision * recall / (precision + recall + 1e-9)
    return {"precision": precision, "recall": recall, "f1": f1}


def parse_csv_list(raw: str) -> list[str]:
    return [v.strip() for v in raw.split(",") if v.strip()]


def normalise_class_names(names_node) -> list[str]:
    if isinstance(names_node, dict):
        return [str(names_node[k]) for k in sorted(names_node, key=lambda x: int(x))]
    if isinstance(names_node, list):
        return [str(v) for v in names_node]
    return []


def resolve_split_path(data_yaml: Path, split_value: str | None, fallback: Path) -> Path:
    if not split_value:
        return fallback.resolve()

    split_path = Path(split_value)
    if not split_path.is_absolute():
        split_path = (data_yaml.parent / split_value).resolve()
    return split_path


def resolve_test_labels_path(data_yaml: Path, test_images: Path) -> Path:
    candidates: list[Path] = []

    # Roboflow layout A: root/test/images -> root/test/labels
    if test_images.name == "images":
        candidates.append((test_images.parent / "labels").resolve())

    # Alternate YOLO layout B: root/images/test -> root/labels/test
    if test_images.parent.name == "images":
        candidates.append((test_images.parent.parent / "labels" / test_images.name).resolve())

    candidates.append((data_yaml.parent / "test" / "labels").resolve())
    candidates.append((data_yaml.parent / "labels" / "test").resolve())

    for path in candidates:
        if path.exists():
            return path

    return candidates[0]


def load_dataset_layout(data_yaml: Path) -> tuple[list[str], Path, Path]:
    with data_yaml.open("r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f) or {}

    class_names = normalise_class_names(data_cfg.get("names", []))
    test_images = resolve_split_path(
        data_yaml,
        data_cfg.get("test"),
        data_yaml.parent / "test" / "images",
    )
    test_labels = resolve_test_labels_path(data_yaml, test_images)
    return class_names, test_images, test_labels


def image_has_defect_gt(label_path: Path, defect_class_ids: set[int]) -> bool:
    """Ground truth: True if the label file contains at least one defect class id."""
    if not label_path.exists() or label_path.stat().st_size == 0:
        return False

    with label_path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            try:
                cls_id = int(float(parts[0]))
            except ValueError:
                continue
            if cls_id in defect_class_ids:
                return True

    return False


def resolve_result_class_name(result, cls_id: int) -> str:
    names = result.names
    if isinstance(names, dict):
        return str(names.get(cls_id, cls_id))
    if isinstance(names, list) and 0 <= cls_id < len(names):
        return str(names[cls_id])
    return str(cls_id)


def image_has_defect_pred(result, conf_thresh: float, defect_class_names: set[str]) -> bool:
    """Prediction: True if any defect-class detection exceeds the confidence threshold."""
    if result.boxes is None or len(result.boxes) == 0:
        return False

    for conf_raw, cls_raw in zip(result.boxes.conf.tolist(), result.boxes.cls.tolist()):
        conf = float(conf_raw)
        if conf < conf_thresh:
            continue

        cls_name = resolve_result_class_name(result, int(cls_raw))
        if cls_name in defect_class_names:
            return True

    return False


def extract_yolo_summary(metrics) -> dict:
    box = getattr(metrics, "box", None)
    if box is None:
        return {}

    summary = {}
    for out_key, attr in {
        "precision": "mp",
        "recall": "mr",
        "mAP50": "map50",
        "mAP50_95": "map",
    }.items():
        value = getattr(box, attr, None)
        if value is not None:
            summary[out_key] = float(value)

    return summary


# ─────────────────────────────────────────────────────────────────────────────
# YOLO-LEVEL EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def run_yolo_validation(model: YOLO, data_yaml: str, conf: float, iou: float) -> dict:
    """
    Run the Ultralytics built-in validation pipeline on the test split.
    Returns bounding-box level metrics (mAP, precision, recall per class).
    """
    print("\n[1/3] Running YOLO validation on test split...")
    metrics = model.val(
        data=data_yaml,
        split="test",
        conf=conf,
        iou=iou,
        save=True,
        save_json=True,
        plots=True,
        project="runs/detect",
        name="bottle_cap_eval",
        verbose=True,
    )
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# IMAGE-LEVEL BINARY CLASSIFICATION EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def run_binary_eval(
    model: YOLO,
    test_images: Path,
    test_labels: Path,
    conf: float,
    iou: float,
    defect_class_ids: set[int],
    defect_class_names: set[str],
) -> dict:
    """
    Beyond bounding-box mAP, evaluate at the image level:
    - Was the image correctly flagged as 'defective' or 'normal'?

    This is the metric that matters most for triage / pass-fail decisions.
    """
    print("\n[2/3] Running image-level binary defect classification eval...")

    tp = fp = tn = fn = 0
    failed_images = []

    image_files = sorted(test_images.glob("*.*"))
    image_files = [f for f in image_files if f.suffix.lower() in IMAGE_EXTENSIONS]

    if not image_files:
        raise FileNotFoundError(f"No evaluation images found in: {test_images}")

    for img_file in image_files:
        lbl_file = test_labels / (img_file.stem + ".txt")
        gt_defective   = image_has_defect_gt(lbl_file, defect_class_ids)
        results        = model.predict(str(img_file), conf=conf, iou=iou, verbose=False)
        pred_defective = image_has_defect_pred(results[0], conf, defect_class_names)

        if gt_defective and pred_defective:
            tp += 1
        elif not gt_defective and pred_defective:
            fp += 1
            failed_images.append({"file": str(img_file), "error": "False Positive"})
        elif gt_defective and not pred_defective:
            fn += 1
            failed_images.append({"file": str(img_file), "error": "False Negative (MISSED)"})
        else:
            tn += 1

    total   = tp + fp + tn + fn
    accuracy = (tp + tn) / (total + 1e-9)
    metrics  = compute_binary_metrics(tp, fp, fn)

    print(f"\n  Image-level Binary Classification Results:")
    print(f"  ┌─────────────────────────────────────────┐")
    print(f"  │  Total images   : {total:>6}               │")
    print(f"  │  True Positive  : {tp:>6}  (correctly flagged defects)    │")
    print(f"  │  True Negative  : {tn:>6}  (correctly passed normals)     │")
    print(f"  │  False Positive : {fp:>6}  (normal flagged as defective)  │")
    print(f"  │  False Negative : {fn:>6}  (MISSED defects ← critical!)   │")
    print(f"  ├─────────────────────────────────────────┤")
    print(f"  │  Accuracy       : {accuracy:.4f}                          │")
    print(f"  │  Precision      : {metrics['precision']:.4f}              │")
    print(f"  │  Recall         : {metrics['recall']:.4f}  ← optimise this  │")
    print(f"  │  F1 Score       : {metrics['f1']:.4f}                     │")
    print(f"  └─────────────────────────────────────────┘")

    if fn > 0:
        print(f"\n  ⚠️  {fn} missed defects (False Negatives).")
        print(f"     → Consider lowering --conf threshold or adding more defect training data.")

    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "accuracy": accuracy,
        **metrics,
        "failed_images": failed_images,
    }


# ─────────────────────────────────────────────────────────────────────────────
# INFERENCE SPEED BENCHMARK
# ─────────────────────────────────────────────────────────────────────────────

def benchmark_speed(model: YOLO, test_images: Path, n_samples: int = 20):
    """
    Time inference on a random sample of test images.
    Gives you a realistic FPS estimate for your deployment target.
    """
    import time
    print(f"\n[3/3] Benchmarking inference speed on {n_samples} images...")

    imgs = [p for p in sorted(test_images.glob("*.*")) if p.suffix.lower() in IMAGE_EXTENSIONS][:n_samples]
    if not imgs:
        print("  No test images found for benchmark.")
        return

    # Warm-up run
    model.predict(str(imgs[0]), verbose=False)

    latencies = []
    for img in imgs:
        t0 = time.perf_counter()
        model.predict(str(img), verbose=False)
        latencies.append((time.perf_counter() - t0) * 1000)  # ms

    avg_ms = np.mean(latencies)
    p95_ms = np.percentile(latencies, 95)
    fps    = 1000.0 / avg_ms

    print(f"\n  Inference Speed Results:")
    print(f"    Average latency  : {avg_ms:.1f} ms")
    print(f"    P95 latency      : {p95_ms:.1f} ms")
    print(f"    Throughput       : {fps:.1f} FPS")

    if fps >= 30:
        print(f"    ✅ Suitable for real-time inspection (≥30 FPS)")
    elif fps >= 10:
        print(f"    ⚡ Suitable for near-real-time inspection (10–30 FPS)")
    else:
        print(f"    ⚠️  Below real-time — consider yolo11n or INT8 quantisation")

    return {"avg_ms": avg_ms, "p95_ms": p95_ms, "fps": fps}


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate YOLOv11 beverage cap-quality model")
    parser.add_argument("--weights", default="runs/detect/bottle_cap_defect/weights/best.pt",
                        help="Path to trained model weights")
    parser.add_argument("--data",    default=DEFAULT_DATA_YAML,
                        help="Path to data.yaml")
    parser.add_argument("--conf",    default=0.25, type=float,
                        help="Confidence threshold (default: 0.25)")
    parser.add_argument("--iou",     default=0.45, type=float,
                        help="NMS IoU threshold (default: 0.45)")
    parser.add_argument("--device",  default="0",
                        help="Device: '0' for GPU, 'cpu' for CPU")
    parser.add_argument("--defect-classes", default=DEFAULT_DEFECT_CLASSES,
                        help="Comma-separated class names considered defective")
    parser.add_argument("--product-category", default="beverage",
                        choices=["beverage", "food"],
                        help="Inspection product category metadata")
    parser.add_argument("--product-sub-type", default="transparent_bottle",
                        choices=["transparent_bottle", "rigid_can", "flexible_wrapper", "rigid_box"],
                        help="Inspection product sub-type metadata")
    args = parser.parse_args()

    weights = Path(args.weights)
    assert weights.exists(), f"Weights not found: {weights}"

    data_yaml = Path(args.data).expanduser().resolve()
    assert data_yaml.exists(), f"data.yaml not found: {data_yaml}"

    class_names, test_images, test_labels = load_dataset_layout(data_yaml)
    class_name_set = set(class_names)
    defect_class_names = set(parse_csv_list(args.defect_classes))
    if not defect_class_names:
        raise ValueError("--defect-classes cannot be empty")

    unknown = sorted(defect_class_names - class_name_set)
    if unknown:
        raise ValueError(
            f"Unknown defect class names: {unknown}. Available classes: {class_names}"
        )

    defect_class_ids = {i for i, name in enumerate(class_names) if name in defect_class_names}

    if not test_images.exists():
        raise FileNotFoundError(f"Resolved test image path does not exist: {test_images}")

    print(f"\n{'='*55}")
    print("  VisionFood QAI — Model Evaluation")
    print(f"{'='*55}")
    print(f"  Product : {args.product_category} / {args.product_sub_type}")
    print(f"  Data    : {data_yaml}")
    print(f"  Classes : {class_names}")
    print(f"  Defects : {sorted(defect_class_names)}")
    print(f"  TestImg : {test_images}")
    print(f"  TestLbl : {test_labels}")
    print(f"  Weights : {weights}")
    print(f"  Conf    : {args.conf}")
    print(f"  IoU     : {args.iou}")

    model = YOLO(str(weights))
    model.to(args.device)

    # 1. YOLO-native bbox-level evaluation
    yolo_metrics = run_yolo_validation(model, str(data_yaml), args.conf, args.iou)

    # 2. Image-level binary classification evaluation
    binary_metrics = run_binary_eval(
        model,
        test_images,
        test_labels,
        args.conf,
        args.iou,
        defect_class_ids,
        defect_class_names,
    )

    # 3. Inference speed benchmark
    speed_metrics = benchmark_speed(model, test_images)

    # 4. Save summary report
    output_dir = Path("runs/detect/bottle_cap_eval")
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "product_category": args.product_category,
        "product_sub_type": args.product_sub_type,
        "data_yaml": str(data_yaml),
        "class_names": class_names,
        "defect_classes": sorted(defect_class_names),
        "test_images": str(test_images),
        "test_labels": str(test_labels),
        "weights":        str(weights),
        "conf_threshold": args.conf,
        "iou_threshold":  args.iou,
        "yolo_metrics":   extract_yolo_summary(yolo_metrics),
        "binary_metrics": binary_metrics,
        "speed":          speed_metrics,
    }
    report_path = output_dir / "eval_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\n✅ Evaluation complete.")
    print(f"   Full report  → {report_path}")
    print(f"   Plots        → runs/detect/bottle_cap_eval/\n")


if __name__ == "__main__":
    main()