"""
threshold_sweep.py
==================
Sweep confidence thresholds for bottle-cap defect screening.

Goal:
    Minimize False Positives while keeping False Negatives at 0.

Example:
    python threshold_sweep.py \
      --weights runs/detect/bottle_cap_defect_quality/weights/best.pt \
      --test-images dataset/Beverages/bottleDefect.v1-first.yolov11-cap/test/images \
      --test-labels dataset/Beverages/bottleDefect.v1-first.yolov11-cap/test/labels \
      --defect-classes defectCap,noCap \
      --device cuda:0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ultralytics import YOLO


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
DEFAULT_WEIGHTS = "runs/detect/bottle_cap_defect_quality/weights/best.pt"
DEFAULT_TEST_IMAGES = "dataset/Beverages/bottleDefect.v1-first.yolov11-cap/test/images"
DEFAULT_TEST_LABELS = "dataset/Beverages/bottleDefect.v1-first.yolov11-cap/test/labels"
DEFAULT_DEFECT_CLASSES = "defectCap,noCap"
DEFAULT_OUT_PATH = "runs/detect/threshold_sweep.json"


def parse_csv_list(raw: str) -> list[str]:
    return [v.strip() for v in raw.split(",") if v.strip()]


def normalize_names(names_node) -> list[str]:
    if isinstance(names_node, dict):
        return [str(names_node[k]) for k in sorted(names_node, key=lambda x: int(x))]
    if isinstance(names_node, list):
        return [str(v) for v in names_node]
    return []


def normalize_device(device: str) -> str:
    if device.strip() == "0":
        return "cuda:0"
    return device


def resolve_result_class_name(result, cls_id: int) -> str:
    names = result.names
    if isinstance(names, dict):
        return str(names.get(cls_id, cls_id))
    if isinstance(names, list) and 0 <= cls_id < len(names):
        return str(names[cls_id])
    return str(cls_id)


def read_label_class_ids(label_path: Path) -> list[int]:
    if not label_path.exists() or label_path.stat().st_size == 0:
        return []

    class_ids: list[int] = []
    with label_path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            try:
                class_ids.append(int(float(parts[0])))
            except ValueError:
                continue

    return class_ids


def image_has_defect_gt(label_path: Path, defect_class_ids: set[int]) -> bool:
    return any(cls_id in defect_class_ids for cls_id in read_label_class_ids(label_path))


def image_has_defect_pred(result, conf_thresh: float, defect_class_names: set[str]) -> bool:
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


def metrics_from_counts(tp: int, fp: int, tn: int, fn: int) -> dict:
    total = tp + fp + tn + fn
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    accuracy = (tp + tn) / (total + 1e-9)
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
    }


def evaluate_at_threshold(
    model: YOLO,
    image_paths: list[Path],
    test_labels: Path,
    conf_thresh: float,
    iou: float,
    defect_class_ids: set[int],
    defect_class_names: set[str],
    device: str,
) -> dict:
    tp = fp = tn = fn = 0

    for img_path in image_paths:
        lbl_path = test_labels / f"{img_path.stem}.txt"
        gt_defect = image_has_defect_gt(lbl_path, defect_class_ids)

        results = model.predict(
            source=str(img_path),
            conf=conf_thresh,
            iou=iou,
            device=device,
            verbose=False,
        )
        pred_defect = image_has_defect_pred(results[0], conf_thresh, defect_class_names)

        if gt_defect and pred_defect:
            tp += 1
        elif gt_defect and not pred_defect:
            fn += 1
        elif not gt_defect and pred_defect:
            fp += 1
        else:
            tn += 1

    row = metrics_from_counts(tp, fp, tn, fn)
    row["conf"] = round(conf_thresh, 4)
    return row


def build_thresholds(min_conf: float, max_conf: float, step: float) -> list[float]:
    thresholds: list[float] = []
    current = min_conf
    while current <= (max_conf + 1e-12):
        thresholds.append(round(current, 4))
        current += step
    return thresholds


def choose_recommendation(rows: list[dict]) -> dict | None:
    eligible = [r for r in rows if r["fn"] == 0]
    if not eligible:
        return None

    # Primary objective: minimize FP with zero FN. Secondary: maximize precision, then confidence.
    eligible.sort(key=lambda r: (r["fp"], -r["precision"], -r["conf"]))
    return eligible[0]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep confidence thresholds and recommend a defect-detection operating point"
    )
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS, help="Path to model weights")
    parser.add_argument("--test-images", default=DEFAULT_TEST_IMAGES, help="Path to test images")
    parser.add_argument("--test-labels", default=DEFAULT_TEST_LABELS, help="Path to test labels")
    parser.add_argument("--defect-classes", default=DEFAULT_DEFECT_CLASSES,
                        help="Comma-separated class names treated as defects")
    parser.add_argument("--min-conf", type=float, default=0.15, help="Start confidence")
    parser.add_argument("--max-conf", type=float, default=0.80, help="End confidence")
    parser.add_argument("--step", type=float, default=0.05, help="Confidence step")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU")
    parser.add_argument("--device", default="cuda:0", help="Device, e.g. cuda:0, 0, cpu")
    parser.add_argument("--out", default=DEFAULT_OUT_PATH, help="Output JSON path")
    args = parser.parse_args()

    weights = Path(args.weights)
    test_images = Path(args.test_images)
    test_labels = Path(args.test_labels)
    out_path = Path(args.out)
    device = normalize_device(args.device)

    if not weights.exists():
        raise FileNotFoundError(f"Weights not found: {weights}")
    if not test_images.exists():
        raise FileNotFoundError(f"Test images path not found: {test_images}")
    if not test_labels.exists():
        raise FileNotFoundError(f"Test labels path not found: {test_labels}")
    if args.step <= 0:
        raise ValueError("--step must be > 0")
    if args.max_conf < args.min_conf:
        raise ValueError("--max-conf must be >= --min-conf")

    model = YOLO(str(weights))
    model.to(device)

    model_class_names = normalize_names(model.names)
    defect_class_names = set(parse_csv_list(args.defect_classes))
    if not defect_class_names:
        raise ValueError("--defect-classes cannot be empty")

    unknown = sorted(defect_class_names - set(model_class_names))
    if unknown:
        raise ValueError(
            f"Unknown defect class names: {unknown}. Available classes: {model_class_names}"
        )

    defect_class_ids = {i for i, n in enumerate(model_class_names) if n in defect_class_names}

    image_paths = sorted(
        p for p in test_images.glob("*.*") if p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_paths:
        raise FileNotFoundError(f"No test images found in: {test_images}")

    thresholds = build_thresholds(args.min_conf, args.max_conf, args.step)

    print(f"\nRunning threshold sweep on {len(image_paths)} test images")
    print(f"Defect classes: {sorted(defect_class_names)}")
    print(f"Device: {device}\n")

    rows: list[dict] = []
    for conf in thresholds:
        row = evaluate_at_threshold(
            model=model,
            image_paths=image_paths,
            test_labels=test_labels,
            conf_thresh=conf,
            iou=args.iou,
            defect_class_ids=defect_class_ids,
            defect_class_names=defect_class_names,
            device=device,
        )
        rows.append(row)

    print(f"{'Conf':>6} | {'TP':>4} {'FP':>4} {'TN':>4} {'FN':>4} | "
          f"{'Precision':>9} {'Recall':>7} {'F1':>7} {'Accuracy':>9}")
    print("-" * 68)
    for r in rows:
        note = "  <- misses defects" if r["fn"] > 0 else ""
        print(
            f"{r['conf']:>6.2f} | {r['tp']:>4} {r['fp']:>4} {r['tn']:>4} {r['fn']:>4} | "
            f"{r['precision']:>9.4f} {r['recall']:>7.4f} {r['f1']:>7.4f} {r['accuracy']:>9.4f}{note}"
        )

    recommendation = choose_recommendation(rows)

    print("\n" + "-" * 68)
    if recommendation is None:
        print("No threshold achieved FN=0. Add defect examples or reduce class confusion first.")
    else:
        print(f"Recommended confidence: {recommendation['conf']:.2f}")
        print(
            f"At conf={recommendation['conf']:.2f}: "
            f"TP={recommendation['tp']} FP={recommendation['fp']} "
            f"TN={recommendation['tn']} FN={recommendation['fn']} "
            f"Precision={recommendation['precision']:.4f} Recall={recommendation['recall']:.4f}"
        )

    payload = {
        "weights": str(weights),
        "test_images": str(test_images),
        "test_labels": str(test_labels),
        "device": device,
        "defect_classes": sorted(defect_class_names),
        "thresholds": rows,
        "recommendation": recommendation,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Full sweep JSON saved to: {out_path}")


if __name__ == "__main__":
    main()