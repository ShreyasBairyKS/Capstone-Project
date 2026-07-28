"""
test_roi_classifiers.py
───────────────────────
End-to-end test: for every image in dataset/NSC (good + bad),
runs the YOLO ROI detector then passes each ROI crop to its
corresponding EfficientAD PyTorch classifier (.ckpt checkpoint).

No ONNX Runtime required — uses PyTorch + anomalib directly.

Outputs:
  - Console summary table (per image) with recall / precision / F1
  - results/test_report.json    — machine-readable results
  - results/heatmaps/           — annotated PNG per image (--save-images)

Usage (from project root on the remote PC):
    python himalaya_label_detection/scripts/test_roi_classifiers.py

    # Save annotated images:
    python himalaya_label_detection/scripts/test_roi_classifiers.py --save-images

    # Only bad images:
    python himalaya_label_detection/scripts/test_roi_classifiers.py --subset bad

    # Isolate one ROI (great for debugging ROI_3):
    python himalaya_label_detection/scripts/test_roi_classifiers.py --roi ROI_3

Prerequisites (all already installed from training):
    ultralytics  anomalib==1.1.0  lightning  timm  torch  opencv-python
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
YOLO_MODELS   = PROJECT_ROOT / "models" / "rois" / "yolo"
MODELS_DIR    = PROJECT_ROOT / "models" / "rois"
THRESHOLD_CFG = PROJECT_ROOT / "himalaya_label_detection" / "config" / "roi_thresholds.json"
RESULTS_DIR   = PROJECT_ROOT / "results"
IMAGE_SIZE    = 256   # must match training

GREEN  = (0, 200, 0)
RED    = (0, 0, 220)
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

    hits = sorted(YOLO_MODELS.rglob("best.pt"), key=lambda p: p.stat().st_mtime)
    if not hits:
        sys.exit(f"❌  No YOLO best.pt found under {YOLO_MODELS}")
    w = hits[-1]
    print(f"  [YOLO] Weights: {w.relative_to(PROJECT_ROOT)}")
    return YOLO(str(w))


def load_pytorch_models() -> Dict[str, Tuple]:
    """
    Load EfficientAD from .ckpt checkpoints saved during training.
    Prefers best.ckpt → last.ckpt → newest .ckpt in the weights dir.
    """
    try:
        import torch
        from anomalib.models.image.efficient_ad.lightning_model import EfficientAd
    except ImportError as exc:
        print(f"  [ERROR] {exc}")
        print("  These should already be installed from training. Check your venv.")
        sys.exit(1)

    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    print(f"  [PyTorch] Device: {device}")

    models = {}
    for roi_name in ["ROI_1", "ROI_2", "ROI_3", "ROI_4"]:
        weights_dir = MODELS_DIR / roi_name / "weights"
        ckpt = None
        for name in ["best.ckpt", "last.ckpt"]:
            c = weights_dir / name
            if c.exists():
                ckpt = c
                break
        if ckpt is None:
            hits = list(weights_dir.rglob("*.ckpt"))
            if hits:
                ckpt = sorted(hits, key=lambda p: p.stat().st_mtime)[-1]
        if ckpt is None:
            print(f"  [SKIP] {roi_name}: no .ckpt found in {weights_dir}")
            continue

        try:
            model = EfficientAd.load_from_checkpoint(str(ckpt))
            model.eval()
            model = model.to(device)
            models[roi_name] = (model, device)
            print(f"  [OK]   {roi_name}: {ckpt.name}")
        except Exception as e:
            print(f"  [WARN] {roi_name}: checkpoint load failed — {e}")

    if not models:
        print("❌  No checkpoints found.")
        print(f"   Expected .ckpt files in: {MODELS_DIR}/ROI_X/weights/")
        sys.exit(1)

    return models


def load_thresholds() -> Dict[str, float]:
    if not THRESHOLD_CFG.exists():
        print(f"  [WARN] roi_thresholds.json not found — using 0.5 for all ROIs")
        return {f"ROI_{i}": 0.5 for i in range(1, 5)}
    raw = json.loads(THRESHOLD_CFG.read_text())
    thresholds = raw.get("thresholds", raw)
    thresholds = {k: v for k, v in thresholds.items() if not k.startswith("_")}
    print(f"  [CFG]  Thresholds: {thresholds}")
    return thresholds


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────

def score_crop(model_device: Tuple, crop_bgr: np.ndarray) -> Tuple[float, Optional[np.ndarray]]:
    """Score one BGR crop. Returns (image_level_score, anomaly_map or None)."""
    import torch
    import torchvision.transforms.functional as TF
    from PIL import Image as PILImage

    model, device = model_device

    rgb    = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    pil    = PILImage.fromarray(rgb)
    tensor = TF.to_tensor(TF.resize(pil, [IMAGE_SIZE, IMAGE_SIZE]))
    tensor = TF.normalize(tensor, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    with torch.no_grad():
        out = model({"image": tensor.unsqueeze(0).to(device)})

    # Extract scalar anomaly score
    if "pred_score" in out:
        score = float(out["pred_score"].squeeze().item())
    elif "anomaly_map" in out:
        score = float(out["anomaly_map"].squeeze().max().item())
    else:
        # Fallback: max of whatever tensor is returned
        for v in out.values():
            try:
                score = float(v.squeeze().max().item())
                break
            except Exception:
                score = 0.5

    amap = out["anomaly_map"].squeeze().cpu().numpy() if "anomaly_map" in out else None
    return score, amap


def yolo_detect_rois(
    model,
    image_bgr: np.ndarray,
    imgsz: int = 1024,
    conf: float = 0.25,
) -> Dict[str, Tuple[int,int,int,int]]:
    """
    Run YOLO on the full image at training resolution (imgsz=1024).
    Returns {ROI_name: (x1,y1,x2,y2)} keeping highest-confidence box per class.

    IMPORTANT: imgsz MUST match the training imgsz (1024).
    At 640 the 8000px tall image gets squished so small that all ROIs disappear.
    """
    results = model(image_bgr, verbose=False, imgsz=imgsz, conf=conf)[0]
    best_conf: Dict[str, float] = {}
    best_box:  Dict[str, Tuple] = {}
    for box in results.boxes:
        cls_id = int(box.cls[0].item())
        name   = results.names[cls_id].upper()   # roi_1 → ROI_1
        if not name.startswith("ROI_"):
            name = f"ROI_{cls_id + 1}"
        c    = float(box.conf[0].item())
        xyxy = tuple(int(v) for v in box.xyxy[0].tolist())
        if c > best_conf.get(name, -1.0):
            best_conf[name] = c
            best_box[name]  = xyxy
    return best_box



# ─────────────────────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────────────────────

def draw_result(
    image_bgr: np.ndarray,
    roi_boxes:  Dict[str, Tuple],
    roi_scores: Dict[str, float],
    thresholds: Dict[str, float],
    overall_pass: bool,
    downscale: int = 4,
) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    vis  = cv2.resize(image_bgr, (w // downscale, h // downscale))

    for roi_name, (x1, y1, x2, y2) in roi_boxes.items():
        score  = roi_scores.get(roi_name)
        thresh = thresholds.get(roi_name, 0.5)
        sx1, sy1, sx2, sy2 = x1//downscale, y1//downscale, x2//downscale, y2//downscale

        if score is None:
            color, label = GREY,  f"{roi_name}: no model"
        elif score > thresh:
            color, label = RED,   f"{roi_name}: {score:.3f} FAIL"
        else:
            color, label = GREEN, f"{roi_name}: {score:.3f} OK"

        cv2.rectangle(vis, (sx1, sy1), (sx2, sy2), color, 2)
        cv2.putText(vis, label, (sx1+4, sy1+18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    verdict = "PASS" if overall_pass else "FAIL"
    col     = GREEN if overall_pass else RED
    cv2.rectangle(vis, (0, 0), (220, 32), col, -1)
    cv2.putText(vis, verdict, (6, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.85, WHITE, 2)
    return vis


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run_test(args: argparse.Namespace) -> None:
    print("\n" + "─"*64)
    print("  ROI Classifier End-to-End Test  (PyTorch .ckpt inference)")
    print("─"*64)

    import torch
    torch.set_float32_matmul_precision("high")  # use Tensor Cores on A5000

    print("\nLoading YOLO detector...")
    yolo = load_yolo()

    print("\nLoading EfficientAD classifiers (.ckpt)...")
    models     = load_pytorch_models()
    thresholds = load_thresholds()

    if args.roi:
        roi_filter = args.roi.upper()
        models     = {k: v for k, v in models.items() if k == roi_filter}
        thresholds = {k: v for k, v in thresholds.items() if k == roi_filter}
        if not models:
            sys.exit(f"❌  ROI '{roi_filter}' not found.")

    # ── Locate dataset ────────────────────────────────────────────
    dataset_dir = resolve_dataset(args.dataset)
    good_dir    = dataset_dir / "NSC GOOD IMAGES"
    bad_dir     = dataset_dir / "NSC BAD IMAGES"
    exts        = {".bmp", ".png", ".jpg", ".jpeg"}

    for d in [good_dir, bad_dir]:
        if not d.exists():
            print(f"❌  Folder not found: {d}")
            print(f"   Pass the correct path with:  --dataset <path to NSC folder>")
            sys.exit(1)

    # ── Gather images ─────────────────────────────────────────────
    image_paths: List[Tuple[Path, str]] = []
    if args.subset in ("good", "all"):
        image_paths += [(p, "good") for p in sorted(good_dir.iterdir()) if p.suffix.lower() in exts]
    if args.subset in ("bad", "all"):
        image_paths += [(p, "bad")  for p in sorted(bad_dir.iterdir())  if p.suffix.lower() in exts]

    if not image_paths:
        sys.exit(f"❌  No images found in {dataset_dir}")

    print(f"\n  {len(image_paths)} images  ({args.subset})  |  "
          f"ROIs being tested: {sorted(models)}\n")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    heatmap_dir = RESULTS_DIR / "heatmaps"
    if args.save_images:
        heatmap_dir.mkdir(parents=True, exist_ok=True)

    # ── Test loop ──────────────────────────────────────────────────
    all_results  = []
    total_correct = 0
    confusion     = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}

    cols   = sorted(models)
    header = f"{'Image':<22} {'True':<5} {'Pred':<5}  " + \
             "  ".join(f"{c:<20}" for c in cols)
    print(header)
    print("─" * max(len(header), 60))

    for img_path, true_label in image_paths:
        t0 = time.perf_counter()

        image_bgr = cv2.imread(str(img_path))
        if image_bgr is None:
            print(f"  [WARN] Cannot read {img_path.name}")
            continue

        roi_boxes = yolo_detect_rois(yolo, image_bgr)

        # Print detection summary for first image only (debugging)
        if len(all_results) == 0:
            if roi_boxes:
                print(f"  [DEBUG] First image YOLO detections: {list(roi_boxes.keys())}")
            else:
                print(f"  [DEBUG] First image: YOLO found NOTHING. Image shape: {image_bgr.shape}")
                print(f"          If shape is very large, YOLO may need a larger imgsz.")

        roi_scores: Dict[str, float] = {}
        roi_amaps:  Dict[str, np.ndarray] = {}

        for roi_name, md in models.items():
            if roi_name not in roi_boxes:
                continue
            x1, y1, x2, y2 = roi_boxes[roi_name]
            crop = image_bgr[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            s, amap = score_crop(md, crop)
            roi_scores[roi_name] = s
            if amap is not None:
                roi_amaps[roi_name] = amap

        failed_rois  = [r for r in cols if r in roi_scores and
                        roi_scores[r] > thresholds.get(r, 0.5)]
        overall_pass = len(failed_rois) == 0
        verdict      = "PASS" if overall_pass else "FAIL"
        elapsed_ms   = (time.perf_counter() - t0) * 1000

        predicted_bad = not overall_pass
        actual_bad    = true_label == "bad"
        correct       = predicted_bad == actual_bad
        if correct: total_correct += 1

        if   actual_bad and predicted_bad:     confusion["TP"] += 1
        elif not actual_bad and not predicted_bad: confusion["TN"] += 1
        elif not actual_bad and predicted_bad: confusion["FP"] += 1
        else:                                  confusion["FN"] += 1

        mark     = "✓" if correct else "✗"
        score_str = "  ".join(
            f"{roi_scores[r]:.3f} {'❌' if r in failed_rois else '✅':<15}"
            if r in roi_scores else f"{'no det':<20}"
            for r in cols
        )
        print(f"{img_path.name:<22} {true_label:<5} {verdict:<5}{mark}  {score_str}  {elapsed_ms:.0f}ms")

        if args.save_images:
            vis = draw_result(image_bgr, roi_boxes, roi_scores, thresholds, overall_pass)
            cv2.imwrite(str(heatmap_dir / f"{true_label}_{img_path.stem}.png"), vis)

        all_results.append({
            "image": img_path.name, "true_label": true_label,
            "verdict": verdict, "correct": correct,
            "roi_scores": roi_scores, "failed_rois": failed_rois,
            "elapsed_ms": round(elapsed_ms, 1),
        })

    # ── Summary ────────────────────────────────────────────────────
    n     = len(all_results)
    n_good = sum(1 for r in all_results if r["true_label"] == "good")
    n_bad  = sum(1 for r in all_results if r["true_label"] == "bad")
    tp, tn, fp, fn = confusion["TP"], confusion["TN"], confusion["FP"], confusion["FN"]

    recall    = tp / (tp + fn)   if (tp + fn) > 0 else float("nan")
    precision = tp / (tp + fp)   if (tp + fp) > 0 else float("nan")
    f1        = 2*precision*recall/(precision+recall) if (precision+recall) > 0 else float("nan")
    accuracy  = total_correct / n if n > 0 else float("nan")

    print("\n" + "═"*64)
    print("  RESULTS SUMMARY")
    print("═"*64)
    print(f"  Images:     {n}  ({n_good} good / {n_bad} bad)")
    print(f"  Accuracy:   {accuracy:.1%}  ({total_correct}/{n})")
    print()
    print(f"  TP (bad caught):       {tp}")
    print(f"  TN (good passed):      {tn}")
    print(f"  FP (false alarms):     {fp}")
    print(f"  FN (missed defects):   {fn}  ← most critical")
    print()
    print(f"  Recall    : {recall:.1%}")
    print(f"  Precision : {precision:.1%}")
    print(f"  F1 Score  : {f1:.3f}")

    if fn > 0:
        missed = [r["image"] for r in all_results if r["true_label"]=="bad" and r["verdict"]=="PASS"]
        print(f"\n  ⚠️  MISSED DEFECTS ({fn}): {missed}")
    if fp > 0:
        alarms = [r["image"] for r in all_results if r["true_label"]=="good" and r["verdict"]=="FAIL"]
        print(f"  ℹ️  FALSE ALARMS  ({fp}): {alarms}")

    print("═"*64)

    report = {
        "summary": {"n_images": n, "n_good": n_good, "n_bad": n_bad,
                    "accuracy": round(accuracy, 4), "recall": round(recall, 4),
                    "precision": round(precision, 4), "f1": round(f1, 4),
                    "TP": tp, "TN": tn, "FP": fp, "FN": fn},
        "thresholds": thresholds,
        "results": all_results,
    }
    out = RESULTS_DIR / "test_report.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\n  Report → {out}")
    if args.save_images:
        print(f"  Images → {heatmap_dir}/")


def resolve_dataset(user_path: Optional[str]) -> Path:
    """Find the NSC dataset folder — user override first, then common locations."""
    if user_path:
        p = Path(user_path)
        if not p.exists():
            sys.exit(f"❌  --dataset path does not exist: {p}")
        return p

    # Auto-search common locations
    candidates = [
        PROJECT_ROOT / "dataset" / "NSC",
        PROJECT_ROOT / "dataset",
        PROJECT_ROOT / "NSC",
        Path("dataset") / "NSC",
        Path("dataset"),
    ]
    for c in candidates:
        if (c / "NSC GOOD IMAGES").exists():
            print(f"  [DATA]  Found dataset at: {c}")
            return c

    print("❌  Could not auto-locate the NSC dataset.")
    print("   Use:  --dataset \"path/to/NSC\"")
    print("   The folder must contain 'NSC GOOD IMAGES' and 'NSC BAD IMAGES' subfolders.")
    sys.exit(1)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--subset",  default="all", choices=["good","bad","all"])
    p.add_argument("--roi",     default=None,  help="Test only this ROI, e.g. ROI_3")
    p.add_argument("--dataset", default=None,  help="Path to the NSC folder containing 'NSC GOOD IMAGES' and 'NSC BAD IMAGES'")
    p.add_argument("--save-images", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    run_test(parse_args())
