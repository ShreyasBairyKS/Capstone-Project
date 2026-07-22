"""
calibrate_roi_thresholds.py
────────────────────────────
Person B — Day 3 script (runs after annotation is done by Person C).

Calibrates an anomaly score threshold for each trained ROI model by:
  1. Scoring all good crops through each ROI model → baseline distribution.
  2. Scoring all bad crops through each ROI model → anomaly distribution.
  3. Plotting the two histograms side-by-side per ROI.
  4. Finding the threshold that achieves target recall on bad crops.
  5. Saving per-ROI thresholds to config/roi_thresholds.json.

Prerequisites:
  - python scripts/train_all_rois.py  (produces models/rois/<roi_name>/)
  - Person C has placed bad crops in data/rois/<roi_name>/bad/

Usage:
    python scripts/calibrate_roi_thresholds.py
    python scripts/calibrate_roi_thresholds.py --target-recall 0.95 --plot
    python scripts/calibrate_roi_thresholds.py --no-save   # dry-run
"""

from __future__ import annotations

import sys
import json
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.image_utils import list_images, load_image

PROJECT_ROOT    = Path(__file__).resolve().parent.parent
MODELS_ROOT     = PROJECT_ROOT / "models" / "rois"
ROIS_ROOT       = PROJECT_ROOT / "data" / "rois"
THRESHOLDS_OUT  = PROJECT_ROOT / "config" / "roi_thresholds.json"
MODEL_PATHS_FILE = MODELS_ROOT / "roi_model_paths.json"

SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
IMAGE_SIZE    = (256, 256)   # must match what was used in training


# ── Model loader ──────────────────────────────────────────────────────────────

def load_roi_model(model_dir: Path):
    """
    Load a trained EfficientAD or PatchCore model from its checkpoint directory.
    Returns the loaded model, or None if no checkpoint is found.
    """
    try:
        # EfficientAD checkpoint
        ckpt_paths = list(model_dir.rglob("*.ckpt"))
        if not ckpt_paths:
            print(f"  [WARN] No checkpoint in {model_dir}")
            return None

        meta_path = model_dir / "model_meta.json"
        model_type = "efficientad"   # default
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            model_type = meta.get("model_type", "efficientad")

        if model_type == "efficientad":
            from anomalib.models import EfficientAd
            model = EfficientAd.load_from_checkpoint(str(ckpt_paths[0]))
        else:
            from anomalib.models import Patchcore
            model = Patchcore.load_from_checkpoint(str(ckpt_paths[0]))

        model.eval()
        return model

    except Exception as exc:
        print(f"  [ERROR] Could not load model from {model_dir}: {exc}")
        return None


# ── Scoring ───────────────────────────────────────────────────────────────────

def score_crop_folder(
    model,
    folder: Path,
    image_size: tuple[int, int],
    label: str,
) -> list[float]:
    """
    Score all images in a folder. Returns a list of anomaly scores (0.0–1.0).
    """
    import torch
    import torchvision.transforms.functional as TF
    from PIL import Image as PILImage

    image_paths = list_images(folder)
    if not image_paths:
        print(f"  [WARN] No images in {folder}")
        return []

    scores = []
    for img_path in tqdm(image_paths, desc=f"  Scoring {label}", leave=False):
        try:
            pil = PILImage.open(img_path).convert("RGB")
            tensor = TF.to_tensor(TF.resize(pil, list(image_size)))
            tensor = TF.normalize(tensor,
                                  mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])
            batch = {"image": tensor.unsqueeze(0)}

            with torch.no_grad():
                output = model(batch)

            score = float(output["pred_score"].item())
            scores.append(score)

        except Exception as exc:
            tqdm.write(f"    [WARN] {img_path.name}: {exc}")
            scores.append(0.0)

    return scores


# ── Threshold finder ──────────────────────────────────────────────────────────

def find_best_threshold(
    good_scores: list[float],
    bad_scores: list[float],
    target_recall: float = 0.95,
) -> tuple[float, dict]:
    """
    Find the lowest anomaly score threshold that achieves `target_recall`
    on bad images while keeping false positives as low as possible.

    Sweeps from 0.0 to 1.0 in 0.001 steps.

    Returns:
        (threshold, metrics_dict)
    """
    if not bad_scores:
        print("  [WARN] No bad scores — cannot calibrate. Using default threshold 0.5")
        return 0.5, {"threshold": 0.5, "recall": 0.0, "fpr": 0.0, "warning": "no_bad_images"}

    best_threshold = 1.0
    best_metrics   = {"threshold": 1.0, "recall": 0.0, "fpr": 1.0}

    for t in np.arange(0.0, 1.001, 0.005):
        tp = sum(1 for s in bad_scores  if s >  t)
        fp = sum(1 for s in good_scores if s >  t)
        fn = sum(1 for s in bad_scores  if s <= t)
        tn = sum(1 for s in good_scores if s <= t)

        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr    = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        if recall >= target_recall:
            best_threshold = float(t)
            best_metrics   = {
                "threshold":        round(float(t), 4),
                "recall":           round(recall, 4),
                "fpr":              round(fpr, 4),
                "true_positives":   tp,
                "false_negatives":  fn,
                "false_positives":  fp,
                "true_negatives":   tn,
                "n_good":           len(good_scores),
                "n_bad":            len(bad_scores),
            }
            break

    return best_threshold, best_metrics


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_histogram(
    good_scores: list[float],
    bad_scores: list[float],
    threshold: float,
    roi_name: str,
    save_path: Path | None = None,
) -> None:
    """Plot overlapping histograms of good vs bad scores with threshold line."""
    try:
        import matplotlib.pyplot as plt
        bins = np.arange(0.0, 1.05, 0.02)
        plt.figure(figsize=(9, 4))
        plt.hist(good_scores, bins=bins, alpha=0.6, label=f"Good (n={len(good_scores)})", color="green")
        plt.hist(bad_scores,  bins=bins, alpha=0.6, label=f"Bad  (n={len(bad_scores)})", color="red")
        plt.axvline(threshold, color="black", linestyle="--",
                    label=f"Threshold = {threshold:.4f}")
        plt.xlabel("Anomaly Score")
        plt.ylabel("Count")
        plt.title(f"Score Distribution — {roi_name}")
        plt.legend()
        plt.tight_layout()
        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150)
            print(f"  Histogram saved → {save_path}")
        else:
            plt.show()
        plt.close()
    except ImportError:
        print("  [WARN] matplotlib not installed — skipping histogram plot")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate per-ROI anomaly thresholds from good/bad score distributions."
    )
    parser.add_argument(
        "--rois-root", type=Path, default=ROIS_ROOT,
        help=f"Root containing ROI subfolders (default: {ROIS_ROOT})"
    )
    parser.add_argument(
        "--models-root", type=Path, default=MODELS_ROOT,
        help=f"Directory with trained ROI models (default: {MODELS_ROOT})"
    )
    parser.add_argument(
        "--target-recall", type=float, default=0.95,
        help="Target recall on bad images (default: 0.95)"
    )
    parser.add_argument(
        "--plot", action="store_true",
        help="Show/save score distribution histograms per ROI"
    )
    parser.add_argument(
        "--no-save", action="store_true",
        help="Print results without saving roi_thresholds.json"
    )
    parser.add_argument(
        "--output", type=Path, default=THRESHOLDS_OUT,
        help=f"Output JSON path (default: {THRESHOLDS_OUT})"
    )
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  HIMALAYA NSC — PER-ROI THRESHOLD CALIBRATION")
    print("=" * 60)
    print(f"\n  Target recall:  {args.target_recall}")
    print(f"  ROI data root:  {args.rois_root}")
    print(f"  Models root:    {args.models_root}\n")

    # ── Discover ROI models ───────────────────────────────────────────────────
    if not args.models_root.exists():
        print(f"[ERROR] Models root not found: {args.models_root}")
        print("  Run: python scripts/train_all_rois.py  first.")
        sys.exit(1)

    roi_model_dirs = sorted([
        d for d in args.models_root.iterdir()
        if d.is_dir() and d.name != "__pycache__"
    ])

    if not roi_model_dirs:
        print(f"[ERROR] No model directories found in {args.models_root}")
        sys.exit(1)

    # ── Process each ROI ──────────────────────────────────────────────────────
    all_thresholds = {}
    summary_rows   = []

    for model_dir in roi_model_dirs:
        roi_name = model_dir.name
        roi_data_dir = args.rois_root / roi_name
        good_dir = roi_data_dir / "good"
        bad_dir  = roi_data_dir / "bad"

        print(f"\n{'─' * 60}")
        print(f"  ROI: {roi_name}")
        print(f"{'─' * 60}")

        # Load model
        model = load_roi_model(model_dir)
        if model is None:
            print(f"  [SKIP] No model checkpoint found.")
            summary_rows.append({"roi": roi_name, "threshold": "N/A", "recall": "N/A", "fpr": "N/A"})
            continue

        # Score good crops
        if not good_dir.exists():
            print(f"  [WARN] No good/ dir at {good_dir}")
            good_scores = []
        else:
            good_scores = score_crop_folder(model, good_dir, IMAGE_SIZE, "good")

        # Score bad crops
        if not bad_dir.exists() or not list_images(bad_dir):
            print(f"  [WARN] No bad/ dir or no images at {bad_dir}")
            print(f"         → Person C needs to add annotated bad crops here")
            bad_scores = []
        else:
            bad_scores = score_crop_folder(model, bad_dir, IMAGE_SIZE, "bad ")

        if good_scores:
            print(f"  Good: n={len(good_scores)}  mean={np.mean(good_scores):.4f}  "
                  f"p99={np.percentile(good_scores, 99):.4f}")
        if bad_scores:
            print(f"  Bad:  n={len(bad_scores)}  mean={np.mean(bad_scores):.4f}  "
                  f"max={np.max(bad_scores):.4f}")

        # Find threshold
        threshold, metrics = find_best_threshold(good_scores, bad_scores, args.target_recall)
        all_thresholds[roi_name] = {
            "threshold": threshold,
            "metrics":   metrics,
        }

        print(f"\n  → Threshold: {threshold:.4f}  |  "
              f"Recall: {metrics.get('recall', 0.0):.3f}  |  "
              f"FPR: {metrics.get('fpr', 0.0):.3f}")

        summary_rows.append({
            "roi":       roi_name,
            "threshold": threshold,
            "recall":    metrics.get("recall", 0.0),
            "fpr":       metrics.get("fpr", 0.0),
            "n_good":    len(good_scores),
            "n_bad":     len(bad_scores),
        })

        # Plot histogram
        if args.plot:
            hist_path = model_dir / "score_histogram.png"
            plot_histogram(good_scores, bad_scores, threshold, roi_name,
                           save_path=hist_path)

    # ── Save thresholds ───────────────────────────────────────────────────────
    if not args.no_save:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        # Also write a flat version for easy lookup by inference pipeline
        flat_thresholds = {
            roi: data["threshold"]
            for roi, data in all_thresholds.items()
        }
        output_data = {
            "thresholds": flat_thresholds,
            "details":    all_thresholds,
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\n\n[OK] Thresholds saved → {args.output}")

    # ── Final table ───────────────────────────────────────────────────────────
    print("\n\n" + "=" * 60)
    print("  CALIBRATION SUMMARY")
    print("=" * 60)
    header = f"  {'ROI':<30}  {'Threshold':>10}  {'Recall':>8}  {'FPR':>8}  {'n_bad':>6}"
    print(header)
    print("  " + "-" * 56)
    for row in summary_rows:
        t = f"{row['threshold']:.4f}" if isinstance(row['threshold'], float) else row['threshold']
        r = f"{row['recall']:.3f}"    if isinstance(row['recall'],    float) else row['recall']
        f_ = f"{row['fpr']:.3f}"     if isinstance(row['fpr'],        float) else row['fpr']
        nb = str(row.get('n_bad', '-'))
        print(f"  {row['roi']:<30}  {t:>10}  {r:>8}  {f_:>8}  {nb:>6}")

    print("\nNext step:")
    print("  python scripts/run_roi_inference.py --folder dataset/NSC/\"NSC BAD IMAGES\"")


if __name__ == "__main__":
    main()
