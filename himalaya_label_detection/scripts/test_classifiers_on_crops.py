"""
test_classifiers_on_crops.py
─────────────────────────────────────────────────────────────────────────────
Tests the PyTorch anomaly classifiers (EfficientAD) directly on pre-cropped
images, completely bypassing YOLO.

Supports testing both color and grayscale model variants in a single run.
Results are saved to:
    runs/color_roi/ROI_*/tp,tn,fp,fn/
    runs/grayscale_roi/ROI_*/tp,tn,fp,fn/

Usage:
  # Test both color and grayscale models for all ROIs at once:
  python himalaya_label_detection/scripts/test_classifiers_on_crops.py --run-both

  # Test only the grayscale models:
  python himalaya_label_detection/scripts/test_classifiers_on_crops.py --data data/gray_scale_rois --models models/gray_scale_rois

  # Test only color models:
  python himalaya_label_detection/scripts/test_classifiers_on_crops.py --data data/rois --models models/rois

  # Test a specific ROI only:
  python himalaya_label_detection/scripts/test_classifiers_on_crops.py --run-both --roi ROI_1
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torchvision.transforms.functional import to_tensor
from anomalib.models import EfficientAd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Default paths
COLOR_DATA_DIR   = PROJECT_ROOT / "data" / "rois"
COLOR_MODELS_DIR = PROJECT_ROOT / "models" / "rois"
GRAY_DATA_DIR    = PROJECT_ROOT / "data" / "gray_scale_rois"
GRAY_MODELS_DIR  = PROJECT_ROOT / "models" / "gray_scale_rois"
RUNS_DIR         = PROJECT_ROOT / "runs"


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_model(roi_name: str, device: str, models_dir: Path) -> Optional[EfficientAd]:
    ckpt_dir = models_dir / roi_name / "weights"
    if not ckpt_dir.exists():
        print(f"  [{roi_name}] ⚠️  No weights folder at {ckpt_dir} — skipping")
        return None

    # Prefer best.ckpt → last.ckpt → newest .ckpt
    ckpt_path = None
    for name in ["best.ckpt", "last.ckpt"]:
        c = ckpt_dir / name
        if c.exists():
            ckpt_path = c
            break
    if ckpt_path is None:
        hits = sorted(ckpt_dir.glob("*.ckpt"), key=lambda p: p.stat().st_mtime)
        if hits:
            ckpt_path = hits[-1]

    if ckpt_path is None:
        print(f"  [{roi_name}] ⚠️  No .ckpt files found in {ckpt_dir} — skipping")
        return None

    print(f"  [{roi_name}] Loading: {ckpt_path.name}")
    model = EfficientAd.load_from_checkpoint(str(ckpt_path), map_location=device)
    model.eval()
    model.to(device)
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Inference helpers
# ─────────────────────────────────────────────────────────────────────────────

def score_crop(model: EfficientAd, img_bgr: np.ndarray, device: str) -> Tuple[float, np.ndarray]:
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, (256, 256), interpolation=cv2.INTER_AREA)
    tensor = to_tensor(img_rgb).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(tensor)

    if isinstance(out, dict) and "anomaly_map" in out:
        amap = out["anomaly_map"].squeeze().cpu().numpy()
        score = float(amap.mean())
    elif hasattr(out, "anomaly_map"):
        amap = out.anomaly_map.squeeze().cpu().numpy()
        score = float(amap.mean())
    elif isinstance(out, torch.Tensor):
        amap = out.squeeze().cpu().numpy()
        score = float(amap.mean())
    else:
        print(f"[WARN] Unknown output format: {type(out)}")
        return 0.0, np.zeros((256, 256))

    return score, amap


def overlay_heatmap(img_bgr: np.ndarray, amap: np.ndarray) -> np.ndarray:
    """Side-by-side: original crop | heatmap overlay."""
    h, w = img_bgr.shape[:2]
    amap_r = cv2.resize(amap, (w, h), interpolation=cv2.INTER_LINEAR)

    min_val, max_val = amap_r.min(), amap_r.max()
    norm = (amap_r - min_val) / (max_val - min_val + 1e-8)
    heatmap = cv2.applyColorMap(np.uint8(255 * norm), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_bgr, 0.6, heatmap, 0.4, 0)
    return np.hstack((img_bgr, overlay))

# ─────────────────────────────────────────────────────────────────────────────
# Threshold optimisation
# ─────────────────────────────────────────────────────────────────────────────

def find_f1_threshold(
    good_scores: List[float],
    bad_scores: List[float],
    steps: int = 1000,
) -> Tuple[float, float]:
    """
    Sweep the full score range in `steps` increments and return the
    (threshold, F1) pair that maximises F1 on this crop set.
    """
    if not bad_scores:
        return 999.0, 0.0

    all_scores = good_scores + bad_scores
    lo, hi = min(all_scores), max(all_scores)
    candidates = np.linspace(lo, hi, steps)

    best_t, best_f1 = candidates[0], 0.0
    for t in candidates:
        tp = sum(1 for s in bad_scores  if s > t)
        fp = sum(1 for s in good_scores if s > t)
        fn = len(bad_scores) - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)

    return best_t, best_f1



def test_roi(
    roi_name: str,
    device: str,
    data_dir: Path,
    models_dir: Path,
    out_run_dir: Path,
) -> Optional[Dict]:
    """
    Test one ROI classifier on its crops.
    Saves annotated images to:
        out_run_dir/ROI_X/tp|tn|fp|fn/
    Returns summary dict or None if skipped.
    """
    print("\n" + "═" * 64)
    print(f"  Testing: {roi_name}  [{out_run_dir.name}]")
    print("═" * 64)

    roi_dir = data_dir / roi_name
    if not roi_dir.exists():
        print(f"  [SKIP] No data folder at {roi_dir}")
        return None

    good_dir = roi_dir / "good"
    bad_dir  = roi_dir / "bad"

    good_files = (
        sorted(good_dir.glob("*.png")) +
        sorted(good_dir.glob("*.bmp")) +
        sorted(good_dir.glob("*.jpg"))
    )
    bad_files = (
        sorted(bad_dir.glob("*.png")) +
        sorted(bad_dir.glob("*.bmp")) +
        sorted(bad_dir.glob("*.jpg"))
    )

    if not good_files and not bad_files:
        print(f"  [SKIP] No images found in good/ or bad/ under {roi_dir}")
        return None

    print(f"  Found {len(good_files)} good crops, {len(bad_files)} bad crops.")

    model = load_model(roi_name, device, models_dir)
    if model is None:
        return None

    # Score all crops
    results = []
    t0 = time.time()
    for f in good_files:
        img = cv2.imread(str(f))
        if img is not None:
            score, amap = score_crop(model, img, device)
            results.append({"path": f, "true_label": "good", "score": score, "amap": amap, "img": img})

    for f in bad_files:
        img = cv2.imread(str(f))
        if img is not None:
            score, amap = score_crop(model, img, device)
            results.append({"path": f, "true_label": "bad", "score": score, "amap": amap, "img": img})
    t1 = time.time()

    bad_scores  = [r["score"] for r in results if r["true_label"] == "bad"]
    good_scores = [r["score"] for r in results if r["true_label"] == "good"]

    # Find the threshold that maximises F1 across the full score range
    threshold, best_f1 = find_f1_threshold(good_scores, bad_scores)

    print(f"\n  Threshold (F1-optimal): {threshold:.4f}  |  Best F1 = {best_f1:.3f}")

    # Categorize & save annotated images
    out_base = out_run_dir / roi_name
    for sub in ["tp", "tn", "fp", "fn"]:
        (out_base / sub).mkdir(parents=True, exist_ok=True)

    tp = fn = fp = tn = 0
    for r in results:
        is_bad   = r["true_label"] == "bad"
        pred_bad = r["score"] > threshold

        if   is_bad and pred_bad:     cat = "tp"; tp += 1
        elif is_bad and not pred_bad: cat = "fn"; fn += 1
        elif not is_bad and pred_bad: cat = "fp"; fp += 1
        else:                         cat = "tn"; tn += 1

        vis = overlay_heatmap(r["img"], r["amap"])
        col = (0, 0, 255) if pred_bad else (0, 200, 0)
        cv2.putText(vis, f"Score: {r['score']:.3f}",  (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)
        cv2.putText(vis, f"Thresh: {threshold:.3f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.putText(vis, cat.upper(), (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)

        out_path = out_base / cat / f"{r['score']:.3f}_{r['path'].name}"
        cv2.imwrite(str(out_path), vis)

    recall    = tp / len(bad_scores)  if bad_scores        else 0.0
    precision = tp / (tp + fp)        if (tp + fp) > 0     else 0.0
    f1        = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    print("\n  Confusion Matrix:")
    print(f"    TP (bad caught):   {tp}")
    print(f"    FN (missed def):   {fn}")
    print(f"    TN (good passed):  {tn}")
    print(f"    FP (false alarm):  {fp}")
    print(f"\n  Recall:    {recall*100:.1f}%")
    print(f"  Precision: {precision*100:.1f}%")
    print(f"  F1 Score:  {f1:.3f}")
    print(f"  Time:      {((t1-t0)/len(results))*1000:.1f}ms per crop")
    print(f"  Images → {out_base}")

    return {"roi": roi_name, "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "recall": recall, "precision": precision, "f1": f1,
            "threshold": threshold}


# ─────────────────────────────────────────────────────────────────────────────
# Print aggregate summary table
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(run_name: str, summaries: List[Dict]) -> None:
    print("\n" + "═" * 72)
    print(f"  SUMMARY — {run_name}")
    print("═" * 72)
    print(f"  {'ROI':<8} {'TP':>4} {'FN':>4} {'TN':>4} {'FP':>4}  {'Recall':>7}  {'Precision':>9}  {'F1':>6}")
    print("  " + "─" * 62)
    for s in summaries:
        print(f"  {s['roi']:<8} {s['tp']:>4} {s['fn']:>4} {s['tn']:>4} {s['fp']:>4}  "
              f"{s['recall']*100:>6.1f}%  {s['precision']*100:>8.1f}%  {s['f1']:>6.3f}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run_one(data_dir: Path, models_dir: Path, run_dir_name: str,
            roi_filter: Optional[str], device: str,
            save_thresholds: bool = False, threshold_cfg: Optional[Path] = None) -> List[Dict]:
    """Run tests for one variant (color or grayscale). Returns summaries."""
    out_run_dir = RUNS_DIR / run_dir_name

    if not models_dir.exists():
        print(f"\n⚠️  Models directory does not exist: {models_dir} — skipping {run_dir_name}")
        return []

    rois_to_test = (
        [roi_filter]
        if roi_filter
        else sorted(d.name for d in models_dir.iterdir() if d.is_dir() and d.name.startswith("ROI_"))
    )

    summaries = []
    for roi in rois_to_test:
        result = test_roi(roi, device, data_dir, models_dir, out_run_dir)
        if result:
            summaries.append(result)

    if summaries:
        print_summary(run_dir_name, summaries)

    if save_thresholds and threshold_cfg and summaries:
        import json as _json
        threshold_cfg.parent.mkdir(parents=True, exist_ok=True)
        existing = {}
        if threshold_cfg.exists():
            try:
                raw = _json.loads(threshold_cfg.read_text())
                existing = raw.get("thresholds", raw)
                existing = {k: v for k, v in existing.items() if not k.startswith("_")}
            except Exception:
                pass
        for s in summaries:
            existing[s["roi"]] = round(s["threshold"], 6)
        threshold_cfg.write_text(_json.dumps(
            {"thresholds": existing,
             "_note": "F1-optimised thresholds from test_classifiers_on_crops.py"},
            indent=2
        ))
        print(f"  ✅ Thresholds saved → {threshold_cfg}")
        for s in summaries:
            print(f"     {s['roi']}: {s['threshold']:.6f}  (F1={s['f1']:.3f})")

    return summaries


def main():
    parser = argparse.ArgumentParser(
        description="Test EfficientAD classifiers on pre-cropped ROI images."
    )
    parser.add_argument("--roi", default=None,
                        help="Test only this ROI, e.g. ROI_1 (default: all)")
    parser.add_argument("--run-both", action="store_true",
                        help="Run both color and grayscale variants in one pass")
    parser.add_argument("--data",   default=None,
                        help="Data root directory (overrides --run-both)")
    parser.add_argument("--models", default=None,
                        help="Models root directory (overrides --run-both)")
    parser.add_argument("--run-name", default=None,
                        help="Output subfolder name under runs/ (default: derived from --models)")
    parser.add_argument("--save-thresholds", action="store_true",
                        help="Write the F1-optimal thresholds to roi_thresholds.json")
    args = parser.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    threshold_cfg = PROJECT_ROOT / "himalaya_label_detection" / "config" / "roi_thresholds.json"

    if args.run_both:
        # ── Run color models ──────────────────────────────────────────────
        print("\n" + "█" * 64)
        print("  COLOR MODELS  (data/rois → models/rois → runs/color_roi)")
        print("█" * 64)
        run_one(COLOR_DATA_DIR, COLOR_MODELS_DIR, "color_roi", args.roi, device)

        # ── Run grayscale models ──────────────────────────────────────────
        print("\n" + "█" * 64)
        print("  GRAYSCALE MODELS  (data/gray_scale_rois → models/gray_scale_rois → runs/grayscale_roi)")
        print("█" * 64)
        run_one(GRAY_DATA_DIR, GRAY_MODELS_DIR, "grayscale_roi", args.roi, device)

    else:
        # ── Single run with explicit paths ────────────────────────────────
        data_dir   = Path(args.data)   if args.data   else COLOR_DATA_DIR
        models_dir = Path(args.models) if args.models else COLOR_MODELS_DIR
        run_name   = args.run_name or models_dir.name
        run_one(data_dir, models_dir, run_name, args.roi, device)


if __name__ == "__main__":
    main()
