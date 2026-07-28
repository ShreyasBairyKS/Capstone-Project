"""
test_roi_classifiers.py
───────────────────────
End-to-end test: for every image in dataset/NSC (good + bad),
runs the YOLO ROI detector then passes each ROI crop to its
corresponding EfficientAD ONNX classifier.

Outputs:
  - Console summary table (per image)
  - results/test_report.json    — machine-readable results
  - results/heatmaps/           — annotated PNG per image (optional, --save-images)

Usage (run on remote PC from project root):
    python himalaya_label_detection/scripts/test_roi_classifiers.py

    # With heatmap images saved:
    python himalaya_label_detection/scripts/test_roi_classifiers.py --save-images

    # Test only bad images:
    python himalaya_label_detection/scripts/test_roi_classifiers.py --subset bad

    # Test only a specific ROI:
    python himalaya_label_detection/scripts/test_roi_classifiers.py --roi ROI_3

Prerequisites:
    pip install ultralytics onnxruntime-gpu opencv-python numpy
    (use onnxruntime instead of onnxruntime-gpu if no CUDA)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

PROJECT_ROOT  = Path(__file__).resolve().parents[2]
YOLO_WEIGHTS  = PROJECT_ROOT / "models" / "rois" / "yolo" / "train4" / "weights" / "best.pt"
MODELS_DIR    = PROJECT_ROOT / "models" / "rois"
THRESHOLD_CFG = PROJECT_ROOT / "himalaya_label_detection" / "config" / "roi_thresholds.json"
DATASET_DIR   = PROJECT_ROOT / "dataset" / "NSC"
RESULTS_DIR   = PROJECT_ROOT / "results"
IMAGE_SIZE    = 256   # must match training


# ─────────────────────────────────────────────────────────────────────────────
# Colour constants (BGR)
# ─────────────────────────────────────────────────────────────────────────────
GREEN  = (0, 200, 0)
RED    = (0, 0, 220)
YELLOW = (0, 200, 220)
WHITE  = (255, 255, 255)
GREY   = (160, 160, 160)


# ─────────────────────────────────────────────────────────────────────────────
# Model loaders
# ─────────────────────────────────────────────────────────────────────────────

def load_yolo():
    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit("❌  pip install ultralytics")

    if not YOLO_WEIGHTS.exists():
        # Try any best.pt under the yolo folder
        hits = list((PROJECT_ROOT / "models" / "rois" / "yolo").rglob("best.pt"))
        if not hits:
            sys.exit(f"❌  YOLO weights not found. Expected: {YOLO_WEIGHTS}")
        w = sorted(hits)[-1]
        print(f"  [YOLO] Using weights: {w}")
        return YOLO(str(w))
    return YOLO(str(YOLO_WEIGHTS))


def load_onnx_sessions() -> Dict[str, object]:
    """Load one ONNX Runtime session per ROI that has a model."""
    try:
        import onnxruntime as ort
    except ImportError as exc:
        print(f"  [ERROR] onnxruntime import failed: {exc}")
        print("  Try: pip uninstall onnxruntime onnxruntime-gpu -y && pip install onnxruntime-gpu")
        sys.exit(1)

    sessions = {}
    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if "CUDAExecutionProvider" in ort.get_available_providers()
        else ["CPUExecutionProvider"]
    )
    print(f"  [ONNX] Providers: {providers}")

    for roi_name in ["ROI_1", "ROI_2", "ROI_3", "ROI_4"]:
        onnx_path = MODELS_DIR / roi_name / "weights" / "model.onnx"
        if onnx_path.exists():
            sess = ort.InferenceSession(str(onnx_path), providers=providers)
            sessions[roi_name] = sess
            inputs = [i.name for i in sess.get_inputs()]
            outputs = [o.name for o in sess.get_outputs()]
            print(f"  [ONNX] {roi_name}: inputs={inputs}  outputs={outputs}")
        else:
            print(f"  [WARN] {roi_name}: ONNX model not found at {onnx_path}")

    return sessions


def load_thresholds() -> Dict[str, float]:
    if not THRESHOLD_CFG.exists():
        print(f"  [WARN] roi_thresholds.json not found — using 0.5 for all ROIs")
        return {f"ROI_{i}": 0.5 for i in range(1, 5)}
    raw = json.loads(THRESHOLD_CFG.read_text())
    thresholds = raw.get("thresholds", raw)
    thresholds = {k: v for k, v in thresholds.items() if not k.startswith("_")}
    print(f"  [CFG] Thresholds: {thresholds}")
    return thresholds


# ─────────────────────────────────────────────────────────────────────────────
# Inference helpers
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_crop(crop_bgr: np.ndarray) -> np.ndarray:
    """Resize and normalise a BGR crop for ONNX input. Returns (1,3,H,W) float32."""
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
    tensor = resized.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    tensor = (tensor - mean) / std
    return tensor.transpose(2, 0, 1)[np.newaxis]  # (1,3,H,W)


def run_onnx(session, crop_bgr: np.ndarray) -> Tuple[float, Optional[np.ndarray]]:
    """
    Run one ONNX session on a crop.
    Returns (image_level_score, anomaly_map_HxW or None).
    Handles both anomalib ONNX export shapes.
    """
    tensor = preprocess_crop(crop_bgr)
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: tensor})
    output_names = [o.name for o in session.get_outputs()]

    score = 0.5
    amap  = None

    for name, out in zip(output_names, outputs):
        arr = np.array(out)
        # anomaly_map is shape (1,1,H,W) or (1,H,W)
        if "map" in name.lower() or arr.ndim >= 3:
            amap = arr.squeeze()
            if amap.ndim == 2:
                score = float(amap.max())  # image-level score = max pixel score
        # pred_score or image-level score is scalar / (1,) / (1,1)
        elif "score" in name.lower() or arr.size == 1:
            score = float(arr.flat[0])

    # If we only got an anomaly map and no scalar score, derive from map
    if amap is not None and score == 0.5:
        score = float(amap.max())

    return score, amap


def yolo_detect_rois(model, image_bgr: np.ndarray) -> Dict[str, Tuple[int, int, int, int]]:
    """
    Run YOLO on the full image.
    Returns {class_name: (x1, y1, x2, y2)} keeping highest-confidence box per class.
    """
    results = model(image_bgr, verbose=False)[0]
    best_conf: Dict[str, float] = {}
    best_box:  Dict[str, Tuple] = {}

    for box in results.boxes:
        cls_id     = int(box.cls[0].item())
        class_name = results.names[cls_id].upper()  # normalise to ROI_1 etc.
        if not class_name.startswith("ROI_"):
            class_name = f"ROI_{cls_id + 1}"
        conf = float(box.conf[0].item())
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]

        if conf > best_conf.get(class_name, -1.0):
            best_conf[class_name] = conf
            best_box[class_name]  = (x1, y1, x2, y2)

    return best_box


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────────────────────

def draw_result(
    image_bgr: np.ndarray,
    roi_boxes: Dict[str, Tuple],
    roi_scores: Dict[str, float],
    thresholds: Dict[str, float],
    overall_pass: bool,
    downscale: int = 4,
) -> np.ndarray:
    """Draw YOLO boxes + anomaly score labels on a downscaled copy."""
    h, w = image_bgr.shape[:2]
    vis = cv2.resize(image_bgr, (w // downscale, h // downscale))

    verdict_color = GREEN if overall_pass else RED
    verdict_text  = "PASS" if overall_pass else "FAIL"

    for roi_name, (x1, y1, x2, y2) in roi_boxes.items():
        score = roi_scores.get(roi_name, None)
        thresh = thresholds.get(roi_name, 0.5)

        sx1, sy1 = x1 // downscale, y1 // downscale
        sx2, sy2 = x2 // downscale, y2 // downscale

        if score is None:
            color = GREY
            label = f"{roi_name}: no model"
        elif score > thresh:
            color = RED
            label = f"{roi_name}: {score:.3f} FAIL (>{thresh:.3f})"
        else:
            color = GREEN
            label = f"{roi_name}: {score:.3f} OK (<={thresh:.3f})"

        cv2.rectangle(vis, (sx1, sy1), (sx2, sy2), color, 2)
        cv2.putText(vis, label, (sx1 + 4, sy1 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    # Overall verdict banner
    cv2.rectangle(vis, (0, 0), (300, 36), verdict_color, -1)
    cv2.putText(vis, f"  {verdict_text}", (4, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, WHITE, 2, cv2.LINE_AA)

    return vis


# ─────────────────────────────────────────────────────────────────────────────
# Main test loop
# ─────────────────────────────────────────────────────────────────────────────

def run_test(args: argparse.Namespace) -> None:
    print("\n" + "─" * 64)
    print("  ROI Classifier End-to-End Test")
    print("─" * 64)

    # ── Load models ───────────────────────────────────────────────
    print("\nLoading YOLO detector...")
    yolo = load_yolo()

    print("\nLoading EfficientAD ONNX classifiers...")
    sessions   = load_onnx_sessions()
    thresholds = load_thresholds()

    # Filter to requested ROI only
    if args.roi:
        roi_filter = args.roi.upper()
        sessions   = {k: v for k, v in sessions.items() if k == roi_filter}
        thresholds = {k: v for k, v in thresholds.items() if k == roi_filter}
        if not sessions:
            sys.exit(f"❌  ROI '{roi_filter}' not found. Available: {list(sessions)}")

    if not sessions:
        sys.exit("❌  No ONNX models found. Check models/rois/ROI_X/weights/model.onnx")

    # ── Gather images ─────────────────────────────────────────────
    good_dir = DATASET_DIR / "NSC GOOD IMAGES"
    bad_dir  = DATASET_DIR / "NSC BAD IMAGES"
    img_exts = {".bmp", ".png", ".jpg", ".jpeg"}

    image_paths: List[Tuple[Path, str]] = []  # (path, "good"|"bad")
    if args.subset in ("good", "all"):
        image_paths += [(p, "good") for p in sorted(good_dir.iterdir())
                        if p.suffix.lower() in img_exts]
    if args.subset in ("bad", "all"):
        image_paths += [(p, "bad") for p in sorted(bad_dir.iterdir())
                        if p.suffix.lower() in img_exts]

    if not image_paths:
        sys.exit(f"❌  No images found in {DATASET_DIR}")

    print(f"\n  Testing on {len(image_paths)} images "
          f"({args.subset}) using ROI(s): {sorted(sessions)}\n")

    # ── Output dirs ───────────────────────────────────────────────
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    heatmap_dir = RESULTS_DIR / "heatmaps"
    if args.save_images:
        heatmap_dir.mkdir(parents=True, exist_ok=True)

    # ── Test loop ─────────────────────────────────────────────────
    all_results = []
    total_correct = 0
    confusion = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}

    header = f"{'Image':<20} {'Label':<6} {'Verdict':<8} " + \
             "  ".join(f"{r:<18}" for r in sorted(sessions))
    print(header)
    print("─" * len(header))

    for img_path, true_label in image_paths:
        t0 = time.perf_counter()

        # Load
        image_bgr = cv2.imread(str(img_path))
        if image_bgr is None:
            print(f"  [WARN] Could not read {img_path.name} — skipping")
            continue

        # YOLO detection
        roi_boxes = yolo_detect_rois(yolo, image_bgr)

        # Score each detected ROI
        roi_scores: Dict[str, float] = {}
        roi_amaps:  Dict[str, np.ndarray] = {}

        for roi_name, sess in sessions.items():
            if roi_name not in roi_boxes:
                continue
            x1, y1, x2, y2 = roi_boxes[roi_name]
            crop = image_bgr[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            score, amap = run_onnx(sess, crop)
            roi_scores[roi_name] = score
            if amap is not None:
                roi_amaps[roi_name] = amap

        # Overall verdict: FAIL if ANY active ROI exceeds its threshold
        active_rois  = [r for r in sessions if r in roi_scores]
        failed_rois  = [r for r in active_rois if roi_scores[r] > thresholds.get(r, 0.5)]
        overall_pass = len(failed_rois) == 0

        elapsed = (time.perf_counter() - t0) * 1000

        # Confusion matrix
        predicted_bad = not overall_pass
        actual_bad    = true_label == "bad"
        if actual_bad and predicted_bad:   confusion["TP"] += 1
        elif not actual_bad and not predicted_bad: confusion["TN"] += 1
        elif not actual_bad and predicted_bad: confusion["FP"] += 1
        elif actual_bad and not predicted_bad: confusion["FN"] += 1
        correct = (predicted_bad == actual_bad)
        if correct:
            total_correct += 1

        # Print row
        verdict = "PASS" if overall_pass else "FAIL"
        score_cols = "  ".join(
            f"{roi_scores.get(r, 'N/A'):>6.3f} {'❌' if r in failed_rois else '✅':<12}"
            if r in roi_scores else f"{'no det':<18}"
            for r in sorted(sessions)
        )
        marker = "✓" if correct else "✗"
        print(f"{img_path.name:<20} {true_label:<6} {verdict:<8}{marker}  {score_cols}  ({elapsed:.0f}ms)")

        # Save visualisation
        if args.save_images:
            vis = draw_result(image_bgr, roi_boxes, roi_scores, thresholds, overall_pass)
            out_name = f"{true_label}_{img_path.stem}_result.png"
            cv2.imwrite(str(heatmap_dir / out_name), vis)

        # Collect result
        all_results.append({
            "image":        img_path.name,
            "true_label":   true_label,
            "verdict":      verdict,
            "correct":      correct,
            "roi_scores":   roi_scores,
            "roi_boxes":    {k: list(v) for k, v in roi_boxes.items()},
            "failed_rois":  failed_rois,
            "elapsed_ms":   round(elapsed, 1),
        })

    # ── Summary ───────────────────────────────────────────────────
    n = len(all_results)
    n_good = sum(1 for r in all_results if r["true_label"] == "good")
    n_bad  = sum(1 for r in all_results if r["true_label"] == "bad")

    tp, tn = confusion["TP"], confusion["TN"]
    fp, fn = confusion["FP"], confusion["FN"]

    recall    = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else float("nan"))
    accuracy  = total_correct / n if n > 0 else float("nan")

    print("\n" + "═" * 64)
    print("  SUMMARY")
    print("═" * 64)
    print(f"  Images tested:  {n}  ({n_good} good  |  {n_bad} bad)")
    print(f"  Accuracy:       {accuracy:.1%}  ({total_correct}/{n} correct)")
    print()
    print(f"  True  Positives (bad caught):     {tp}")
    print(f"  True  Negatives (good passed):    {tn}")
    print(f"  False Positives (good flagged):   {fp}  ← false alarms")
    print(f"  False Negatives (bad missed):     {fn}  ← CRITICAL missed defects")
    print()
    print(f"  Recall    (catch rate):   {recall:.1%}")
    print(f"  Precision (alarm accuracy): {precision:.1%}")
    print(f"  F1 Score:                 {f1:.3f}")
    print()
    print(f"  Per-ROI threshold used:")
    for roi_name, thresh in sorted(thresholds.items()):
        if roi_name in sessions:
            print(f"    {roi_name}: {thresh:.4f}")

    if fn > 0:
        missed = [r["image"] for r in all_results
                  if r["true_label"] == "bad" and r["verdict"] == "PASS"]
        print(f"\n  ⚠️  MISSED BAD IMAGES ({fn}): {missed}")

    if fp > 0:
        false_alarms = [r["image"] for r in all_results
                        if r["true_label"] == "good" and r["verdict"] == "FAIL"]
        print(f"\n  ℹ️  FALSE ALARMS ({fp}): {false_alarms}")

    print("═" * 64)

    # ── Save JSON report ──────────────────────────────────────────
    report = {
        "summary": {
            "n_images": n, "n_good": n_good, "n_bad": n_bad,
            "accuracy": round(accuracy, 4),
            "recall": round(recall, 4),
            "precision": round(precision, 4),
            "f1": round(f1, 4),
            "TP": tp, "TN": tn, "FP": fp, "FN": fn,
        },
        "thresholds": thresholds,
        "results": all_results,
    }
    report_path = RESULTS_DIR / "test_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"\n  Full report saved → {report_path}")
    if args.save_images:
        print(f"  Annotated images  → {heatmap_dir}/")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Test ROI classifiers on the NSC dataset end-to-end"
    )
    p.add_argument("--subset", default="all", choices=["good", "bad", "all"],
                   help="Which images to test (default: all)")
    p.add_argument("--roi", default=None,
                   help="Test only this ROI, e.g. --roi ROI_3 (default: all)")
    p.add_argument("--save-images", action="store_true",
                   help="Save annotated result images to results/heatmaps/")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_test(args)
