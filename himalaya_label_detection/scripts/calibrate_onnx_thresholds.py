"""Calibrate PASS/FAIL thresholds for the ROI ONNX anomaly-map models.

The script uses the exact deployment score:
    score = mean(anomaly_map)

Input data must be arranged as:
    <data-root>/ROI_1/good/*
    <data-root>/ROI_1/bad/*
    ... through ROI_4

Both colour and already-grayscale image files are accepted. Every image is
converted to grayscale before inference, so calibration matches the LabVIEW
grayscale deployment contract. The result is calibration evidence, not a
substitute for evaluation on an independent held-out set.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import onnxruntime as ort


ROI_NAMES = ("ROI_1", "ROI_2", "ROI_3", "ROI_4")
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}


@dataclass(frozen=True)
class ImageScore:
    roi: str
    label: str
    path: str
    score: float


def image_paths(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted(path for path in folder.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def image_to_tensor(path: Path) -> np.ndarray:
    """Apply the required grayscale ONNX preprocessing and return [1,3,256,256]."""
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError("OpenCV could not read the image")
    gray_256 = cv2.resize(gray, (256, 256), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(gray_256, cv2.COLOR_GRAY2RGB)
    return (rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)[None, ...]


def score_images(session: ort.InferenceSession, roi_name: str, label: str, paths: Iterable[Path], data_root: Path) -> list[ImageScore]:
    results: list[ImageScore] = []
    for path in paths:
        try:
            anomaly_map = session.run(["anomaly_map"], {"image": image_to_tensor(path)})[0]
            score = float(np.asarray(anomaly_map, dtype=np.float32)[0, 0].mean())
        except Exception as exc:
            raise RuntimeError(f"{roi_name} {label} image failed: {path} ({exc})") from exc
        if not np.isfinite(score):
            raise RuntimeError(f"{roi_name} {label} image returned a non-finite score: {path}")
        results.append(ImageScore(roi_name, label, str(path.relative_to(data_root)), score))
    return results


def metrics(good_scores: np.ndarray, bad_scores: np.ndarray, threshold: float) -> dict[str, float | int]:
    tp = int(np.count_nonzero(bad_scores > threshold))
    fn = int(bad_scores.size - tp)
    fp = int(np.count_nonzero(good_scores > threshold))
    tn = int(good_scores.size - fp)
    recall = tp / (tp + fn) if tp + fn else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    return {"tp": tp, "fn": fn, "tn": tn, "fp": fp, "recall": recall, "precision": precision, "f1": f1, "false_positive_rate": fpr}


def candidate_thresholds(good_scores: np.ndarray, bad_scores: np.ndarray) -> np.ndarray:
    values = np.unique(np.concatenate((good_scores, bad_scores)))
    values.sort()
    below_minimum = np.nextafter(values[0], -np.inf)
    above_maximum = np.nextafter(values[-1], np.inf)
    if values.size == 1:
        return np.array((below_minimum, values[0], above_maximum), dtype=np.float64)
    between_values = values[:-1] + (values[1:] - values[:-1]) / 2.0
    return np.concatenate(([below_minimum], between_values, [above_maximum]))


def choose_threshold(good_scores: np.ndarray, bad_scores: np.ndarray, method: str, target_recall: float) -> tuple[float, dict[str, float | int]]:
    candidates = candidate_thresholds(good_scores, bad_scores)
    evaluated = [(float(threshold), metrics(good_scores, bad_scores, float(threshold))) for threshold in candidates]
    if method == "target-recall":
        eligible = [(threshold, result) for threshold, result in evaluated if result["recall"] >= target_recall]
        # Highest eligible threshold minimizes false rejections while retaining the requested recall.
        return max(eligible, key=lambda item: item[0])
    # For equal F1, choose the higher threshold, which avoids unnecessary false positives.
    return max(evaluated, key=lambda item: (item[1]["f1"], item[0]))


def parse_args() -> argparse.Namespace:
    package_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="folder containing ROI_1/good, ROI_1/bad, ... ROI_4",
    )
    parser.add_argument(
        "--models-root",
        type=Path,
        default=package_root / "models",
        help="folder containing ROI_N/weights/model.onnx (default: handoff models folder)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=package_root / "config" / "roi_onnx_thresholds.json",
        help="calibrated JSON to write",
    )
    parser.add_argument(
        "--scores-output",
        type=Path,
        default=None,
        help="optional per-image-score CSV; defaults beside --output",
    )
    parser.add_argument(
        "--method",
        choices=("f1", "target-recall"),
        default="f1",
        help="threshold selection method (default: f1)",
    )
    parser.add_argument(
        "--target-recall",
        type=float,
        default=0.95,
        help="required recall when --method target-recall is selected (default: 0.95)",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="write a clearly incomplete config if an ROI model or labelled crop set is missing",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 0.0 < args.target_recall <= 1.0:
        raise SystemExit("--target-recall must be greater than 0 and no greater than 1")

    data_root = args.data_root.resolve()
    models_root = args.models_root.resolve()
    output_path = args.output.resolve()
    scores_path = (args.scores_output or output_path.with_suffix(".scores.csv")).resolve()
    if not data_root.is_dir():
        raise SystemExit(f"Data root does not exist: {data_root}")

    all_scores: list[ImageScore] = []
    thresholds: dict[str, float | None] = {roi: None for roi in ROI_NAMES}
    roi_results: dict[str, dict[str, object]] = {}
    unavailable: list[str] = []

    for roi_name in ROI_NAMES:
        model_path = models_root / roi_name / "weights" / "model.onnx"
        good_paths = image_paths(data_root / roi_name / "good")
        bad_paths = image_paths(data_root / roi_name / "bad")
        if not model_path.is_file() or not good_paths or not bad_paths:
            reason = []
            if not model_path.is_file():
                reason.append("model.onnx missing")
            if not good_paths:
                reason.append("no good images")
            if not bad_paths:
                reason.append("no bad images")
            unavailable.append(f"{roi_name}: {', '.join(reason)}")
            continue

        print(f"Scoring {roi_name}: {len(good_paths)} good, {len(bad_paths)} bad")
        session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        roi_scores = score_images(session, roi_name, "good", good_paths, data_root)
        roi_scores += score_images(session, roi_name, "bad", bad_paths, data_root)
        all_scores.extend(roi_scores)
        good_scores = np.array([record.score for record in roi_scores if record.label == "good"], dtype=np.float64)
        bad_scores = np.array([record.score for record in roi_scores if record.label == "bad"], dtype=np.float64)
        threshold, result = choose_threshold(good_scores, bad_scores, args.method, args.target_recall)
        thresholds[roi_name] = threshold
        roi_results[roi_name] = {
            "n_good": int(good_scores.size),
            "n_bad": int(bad_scores.size),
            "threshold": threshold,
            "good_score_range": [float(good_scores.min()), float(good_scores.max())],
            "bad_score_range": [float(bad_scores.min()), float(bad_scores.max())],
            "calibration_metrics": result,
        }
        print(
            f"  threshold={threshold:.9f}, F1={result['f1']:.3f}, "
            f"recall={result['recall']:.3f}, precision={result['precision']:.3f}, FPR={result['false_positive_rate']:.3f}"
        )

    if unavailable and not args.allow_partial:
        print("\nNo configuration was written. Missing calibration inputs:")
        for message in unavailable:
            print(f"  - {message}")
        print("Provide good and bad crops for every ROI, or use --allow-partial only for diagnostic work.")
        return 2

    status = "CALIBRATED_REQUIRES_HELD_OUT_VALIDATION" if not unavailable else "PARTIALLY_CALIBRATED_NOT_DEPLOYABLE"
    document = {
        "calibration_status": status,
        "score": "mean(anomaly_map)",
        "preprocessing": "uint8 grayscale -> direct 256x256 OpenCV INTER_AREA resize -> replicate RGB -> float32 / 255; no external mean/std normalization",
        "decision_rule": "FAIL when score > threshold; PASS when score <= threshold",
        "selection_method": args.method,
        "target_recall": args.target_recall if args.method == "target-recall" else None,
        "calibration_data_root": str(data_root),
        "thresholds": thresholds,
        "per_roi": roi_results,
        "missing_or_incomplete_rois": unavailable,
        "warning": "Thresholds were selected on calibration data. Confirm performance with an independent held-out labelled set before production use.",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    scores_path.parent.mkdir(parents=True, exist_ok=True)
    with scores_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("roi", "label", "path", "score"))
        writer.writeheader()
        writer.writerows(asdict(record) for record in all_scores)
    print(f"\nWrote thresholds: {output_path}")
    print(f"Wrote calibration scores: {scores_path}")
    if unavailable:
        print("WARNING: This is partial and must not be sent as a deployable four-ROI threshold file.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
