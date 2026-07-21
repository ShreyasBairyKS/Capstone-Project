"""
validate.py
───────────
Phase 7 — Acceptance testing.

Measures recall, precision, false positive rate, F2 score, and latency
against a labelled test set.

Test set layout:
    data/test/
        good/    ← known-good images  (false positive measurement)
        bad/     ← known-defective images  (recall measurement)

Acceptance criteria (from architecture spec):
    Recall        ≥ 0.990  (miss rate < 1%)
    FP rate       ≤ 0.150
    P95 latency   ≤ 50 ms

Usage:
    python scripts/validate.py
    python scripts/validate.py --test-dir data/test/ --plot
"""

import sys
import argparse
import time
import numpy as np
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline import HimalayaLabelInspector
from src.utils.image_utils import load_image, list_images

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR   = PROJECT_ROOT / "config"


def f_beta(precision: float, recall: float, beta: float = 2.0) -> float:
    denom = beta**2 * precision + recall
    if denom == 0:
        return 0.0
    return (1 + beta**2) * precision * recall / denom


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 7 acceptance validation.")
    parser.add_argument("--test-dir", type=Path,
                        default=PROJECT_ROOT / "data" / "test",
                        help="Directory with good/ and bad/ subfolders")
    parser.add_argument("--plot", action="store_true",
                        help="Show confusion matrix and latency histogram")
    args = parser.parse_args()

    good_dir = args.test_dir / "good"
    bad_dir  = args.test_dir / "bad"

    good_paths = list_images(good_dir) if good_dir.exists() else []
    bad_paths  = list_images(bad_dir)  if bad_dir.exists()  else []

    if not good_paths and not bad_paths:
        print(f"[ERROR] No test images found in {args.test_dir}")
        print("  Create data/test/good/ and data/test/bad/ with test images.")
        sys.exit(1)

    print(f"\nTest set: {len(good_paths)} good  |  {len(bad_paths)} bad")
    print("[Loading pipeline...]\n")
    inspector = HimalayaLabelInspector.from_config(CONFIG_DIR)

    latencies = []
    tp = fp = tn = fn = 0

    for img_path in tqdm(good_paths, desc="Testing good"):
        raw = load_image(img_path)
        t0  = time.perf_counter()
        res = inspector.inspect(raw)
        latencies.append((time.perf_counter() - t0) * 1000)
        if res.passed:
            tn += 1     # correctly passed (true negative)
        else:
            fp += 1     # false positive — good label wrongly rejected

    for img_path in tqdm(bad_paths, desc="Testing bad "):
        raw = load_image(img_path)
        t0  = time.perf_counter()
        res = inspector.inspect(raw)
        latencies.append((time.perf_counter() - t0) * 1000)
        if not res.passed:
            tp += 1     # correctly rejected (true positive)
        else:
            fn += 1     # FALSE NEGATIVE — missed defect

    total     = tp + fp + tn + fn
    recall    = tp / (tp + fn)  if (tp + fn)  > 0 else 0.0
    precision = tp / (tp + fp)  if (tp + fp)  > 0 else 0.0
    fpr       = fp / (fp + tn)  if (fp + tn)  > 0 else 0.0
    f2        = f_beta(precision, recall, beta=2.0)
    lat       = np.array(latencies) if latencies else np.array([0.0])

    TARGET_RECALL  = 0.990
    TARGET_FPR     = 0.150
    TARGET_LAT_P95 = 50.0

    print("\n" + "=" * 58)
    print("  ACCEPTANCE TEST RESULTS")
    print("=" * 58)
    print(f"  Total images:          {total}")
    print(f"  True  Positives (TP):  {tp:4d}   defects caught")
    print(f"  False Negatives (FN):  {fn:4d}   defects MISSED  ← critical")
    print(f"  True  Negatives (TN):  {tn:4d}   good labels passed")
    print(f"  False Positives (FP):  {fp:4d}   good labels rejected")
    print()
    passed_recall = recall >= TARGET_RECALL
    passed_fpr    = fpr    <= TARGET_FPR
    passed_lat    = np.percentile(lat, 95) <= TARGET_LAT_P95

    def gate(ok: bool) -> str:
        return "✓" if ok else "✗"

    print(f"  {gate(passed_recall)} Recall:               {recall:.3f}   "
          f"(target ≥ {TARGET_RECALL})")
    print(f"    Precision:            {precision:.3f}")
    print(f"  {gate(passed_fpr)} False Positive Rate:  {fpr:.3f}   "
          f"(target ≤ {TARGET_FPR})")
    print(f"    F2 Score:             {f2:.3f}")
    print()
    print(f"  {gate(passed_lat)} P50 latency:          {np.percentile(lat,50):.1f} ms")
    print(f"  {gate(passed_lat)} P95 latency:          {np.percentile(lat,95):.1f} ms   "
          f"(target ≤ {TARGET_LAT_P95} ms)")
    print(f"    P99 latency:          {np.percentile(lat,99):.1f} ms")
    print("=" * 58)

    overall = passed_recall and passed_fpr and passed_lat
    print(f"\n  Acceptance gate: {'✓  PASSED' if overall else '✗  FAILED'}\n")

    if not overall:
        if not passed_recall:
            print(f"  → Recall {recall:.3f} < {TARGET_RECALL}:")
            print(f"    Lower thresholds:  "
                  f"python scripts/calibrate_thresholds.py --target-recall 0.99")
        if not passed_fpr:
            print(f"  → FP rate {fpr:.3f} > {TARGET_FPR}:")
            print(f"    Collect more good images or raise thresholds")
        if not passed_lat:
            print(f"  → P95 latency {np.percentile(lat,95):.1f} ms > {TARGET_LAT_P95} ms:")
            print(f"    Use GPU, reduce image size in pipeline_config.yaml, "
                  f"or disable LR/GM checks")

    if args.plot:
        try:
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))

            # Confusion matrix
            cm = np.array([[tn, fp], [fn, tp]])
            im = axes[0].imshow(cm, cmap="Blues")
            for i in range(2):
                for j in range(2):
                    axes[0].text(j, i, cm[i, j],
                                 ha="center", va="center",
                                 fontsize=14, color="black")
            axes[0].set_xticks([0, 1])
            axes[0].set_yticks([0, 1])
            axes[0].set_xticklabels(["Pred PASS", "Pred FAIL"])
            axes[0].set_yticklabels(["Actual Good", "Actual Bad"])
            axes[0].set_title("Confusion Matrix")

            # Latency histogram
            axes[1].hist(latencies, bins=25, color="steelblue", edgecolor="white")
            axes[1].axvline(np.percentile(lat, 95), color="red", linestyle="--",
                            label=f"P95 = {np.percentile(lat,95):.1f} ms")
            axes[1].set_xlabel("Latency (ms)")
            axes[1].set_ylabel("Count")
            axes[1].set_title("Inference Latency Distribution")
            axes[1].legend()

            plt.tight_layout()
            plt.show()
        except ImportError:
            print("[WARN] matplotlib not installed — skipping plots")


if __name__ == "__main__":
    main()
