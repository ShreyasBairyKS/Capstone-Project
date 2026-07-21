"""
calibrate_thresholds.py
────────────────────────
Phase 4 utility — find optimal anomaly thresholds using the labelled
(good / bad) split from Phase 1.

Approach:
  1. Score all aligned good images (80) through the pipeline.
  2. Score all aligned bad images (50) through the pipeline.
  3. Sweep thresholds and find the lowest value that achieves the
     target recall on bad images.
  4. Write the result back to config/thresholds.yaml.

Target: recall ≥ 0.99 on bad images (miss rate < 1%).

Usage:
    python scripts/calibrate_thresholds.py
    python scripts/calibrate_thresholds.py --target-recall 0.99 --plot
    python scripts/calibrate_thresholds.py --no-save          # dry-run
"""

import sys
import argparse
import yaml
import numpy as np
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessing.align import LabelAligner
from src.preprocessing.roi_extractor import ROIExtractor
from src.pipeline import AnomalyScorer
from src.utils.image_utils import load_image, list_images

PROJECT_ROOT    = Path(__file__).resolve().parent.parent
PIPELINE_CONFIG = PROJECT_ROOT / "config" / "pipeline_config.yaml"
THRESHOLDS_PATH = PROJECT_ROOT / "config" / "thresholds.yaml"
ROI_CONFIG_PATH = PROJECT_ROOT / "config" / "roi_config.yaml"
GM_PATH         = PROJECT_ROOT / "golden_master" / "golden_master.jpg"


def load_pipeline_config() -> dict:
    with open(PIPELINE_CONFIG) as f:
        return yaml.safe_load(f)


def score_images(
    image_paths: list[Path],
    aligner: LabelAligner,
    roi_extractor: ROIExtractor,
    scorer: AnomalyScorer,
    target_size: tuple,
    label: str,
) -> list[float]:
    """Align → ROI extract → score. Returns the max ROI anomaly score per image."""
    scores = []
    for img_path in tqdm(image_paths, desc=f"Scoring {label}"):
        try:
            raw          = load_image(img_path, target_size=target_size)
            align        = aligner.align(raw)
            roi_result   = roi_extractor.extract(align.image)
            roi_scores   = []
            for roi_name, crop in roi_result.crops.items():
                if roi_result.valid.get(roi_name):
                    s, _ = scorer.score(crop)
                    roi_scores.append(s)
            scores.append(max(roi_scores) if roi_scores else 0.0)
        except Exception as exc:
            tqdm.write(f"  [WARN] {img_path.name}: {exc}")
            scores.append(0.0)
    return scores


def find_threshold(
    good_scores: list[float],
    bad_scores:  list[float],
    target_recall: float = 0.99,
) -> tuple[float, dict]:
    """
    Find the lowest threshold that achieves `target_recall` on bad images.

    Returns:
        (threshold, metrics_dict)
    """
    best_threshold = 1.0
    best_metrics   = {
        "threshold": 1.0, "recall": 0.0, "false_positive_rate": 0.0,
        "true_positives": 0, "false_negatives": len(bad_scores),
        "false_positives": 0, "true_negatives": len(good_scores),
    }

    for t in np.arange(0.0, 1.005, 0.005):
        tp = sum(1 for s in bad_scores  if s >  t)
        fp = sum(1 for s in good_scores if s >  t)
        fn = sum(1 for s in bad_scores  if s <= t)
        tn = sum(1 for s in good_scores if s <= t)

        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr    = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        if recall >= target_recall:
            best_threshold = float(t)
            best_metrics   = {
                "threshold":           round(float(t), 3),
                "recall":              round(recall, 4),
                "false_positive_rate": round(fpr, 4),
                "true_positives":      tp,
                "false_negatives":     fn,
                "false_positives":     fp,
                "true_negatives":      tn,
            }
            break

    return best_threshold, best_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate anomaly thresholds.")
    parser.add_argument("--target-recall", type=float, default=0.99,
                        help="Target recall on bad images (default: 0.99)")
    parser.add_argument("--plot",    action="store_true",
                        help="Show score distribution histogram")
    parser.add_argument("--no-save", action="store_true",
                        help="Print results without updating thresholds.yaml")
    args = parser.parse_args()

    cfg         = load_pipeline_config()
    target_size = (cfg["image"]["width"], cfg["image"]["height"])

    aligner = LabelAligner(
        golden_master_path=GM_PATH,
        target_size=target_size,
        orb_max_features=cfg["alignment"]["orb_max_features"],
        match_keep_top=cfg["alignment"]["match_keep_top"],
        ransac_threshold=cfg["alignment"]["ransac_threshold"],
        min_match_count=cfg["alignment"]["min_match_count"],
    )
    roi_extractor = ROIExtractor(ROI_CONFIG_PATH)
    scorer        = AnomalyScorer(PROJECT_ROOT / "models" / "full_label")

    good_paths = list_images(PROJECT_ROOT / cfg["paths"]["aligned_good"])
    bad_paths  = list_images(PROJECT_ROOT / cfg["paths"]["aligned_bad"])

    if not good_paths or not bad_paths:
        print("[ERROR] No aligned images found. Run Phase 1 first.")
        sys.exit(1)

    print(f"\nScoring {len(good_paths)} good  |  {len(bad_paths)} bad images...\n")
    good_scores = score_images(good_paths, aligner, roi_extractor,
                               scorer, target_size, "good")
    bad_scores  = score_images(bad_paths,  aligner, roi_extractor,
                               scorer, target_size, "bad ")

    print(f"\nGood scores — mean: {np.mean(good_scores):.4f}  "
          f"max: {np.max(good_scores):.4f}")
    print(f"Bad  scores — mean: {np.mean(bad_scores):.4f}  "
          f"max: {np.max(bad_scores):.4f}")

    threshold, metrics = find_threshold(good_scores, bad_scores, args.target_recall)

    print(f"\n── Calibration Result ─────────────────────────────────")
    for k, v in metrics.items():
        print(f"  {k:<30} {v}")

    if args.plot:
        try:
            import matplotlib.pyplot as plt
            bins = np.arange(0, 1.05, 0.025)
            plt.figure(figsize=(9, 4))
            plt.hist(good_scores, bins=bins, alpha=0.6, label="Good",  color="green")
            plt.hist(bad_scores,  bins=bins, alpha=0.6, label="Bad",   color="red")
            plt.axvline(threshold, color="black", linestyle="--",
                        label=f"Threshold = {threshold:.3f}")
            plt.xlabel("Max ROI Anomaly Score")
            plt.ylabel("Count")
            plt.title("Score Distribution — Good vs. Bad Labels")
            plt.legend()
            plt.tight_layout()
            plt.show()
        except ImportError:
            print("[WARN] matplotlib not installed — skipping plot")

    if not args.no_save:
        with open(THRESHOLDS_PATH) as f:
            tcfg = yaml.safe_load(f)
        tcfg["anomaly"]["critical_roi_threshold"] = round(threshold, 3)
        tcfg["anomaly"]["general_roi_threshold"]  = round(
            min(threshold + 0.05, 0.99), 3
        )
        with open(THRESHOLDS_PATH, "w") as f:
            yaml.dump(tcfg, f, default_flow_style=False, sort_keys=False)
        print(f"\n[OK] Thresholds updated in config/thresholds.yaml")
        print(f"     critical = {tcfg['anomaly']['critical_roi_threshold']}")
        print(f"     general  = {tcfg['anomaly']['general_roi_threshold']}")
    else:
        print("\n[--no-save] config/thresholds.yaml NOT modified.")


if __name__ == "__main__":
    main()
