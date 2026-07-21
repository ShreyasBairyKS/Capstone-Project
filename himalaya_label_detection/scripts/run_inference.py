"""
run_inference.py
────────────────
Phase 6 script — run the full inspection pipeline on images.

Usage:
    # Single image
    python scripts/run_inference.py --image path/to/label.jpg

    # Entire folder
    python scripts/run_inference.py --folder data/raw/bad/

    # Show heatmap window on FAIL
    python scripts/run_inference.py --folder data/raw/bad/ --show-heatmap

    # Save annotated output images + JSON results
    python scripts/run_inference.py --folder data/raw/bad/ \\
        --save-output results/heatmaps/ --json-out results/run.json
"""

import sys
import argparse
import json
import time
import cv2
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline import HimalayaLabelInspector
from src.utils.image_utils import load_image, list_images, save_image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR   = PROJECT_ROOT / "config"


def annotate_image(
    image,
    result,
) -> "np.ndarray":
    """Draw verdict banner on the image (green=PASS, red=FAIL)."""
    import numpy as np
    canvas = image.copy()
    color  = (0, 200, 0) if result.passed else (0, 0, 220)
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 40), color, -1)
    cv2.putText(canvas, result.verdict,
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    if not result.passed and result.fusion.reasons:
        reason_text = result.fusion.reasons[0][:80]
        cv2.putText(canvas, reason_text,
                    (10, canvas.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 220), 1)
    return canvas


def run_single(
    inspector,
    image_path: Path,
    show_heatmap: bool = False,
    save_dir: Path = None,
) -> dict:
    raw = load_image(image_path)
    t0  = time.perf_counter()
    result = inspector.inspect(raw)
    latency_ms = (time.perf_counter() - t0) * 1000

    status_char = "✓" if result.passed else "✗"
    print(f"  {status_char} {image_path.name:<40}  {result.verdict}  "
          f"{latency_ms:5.1f} ms  inliers={result.alignment_inliers}")

    if not result.passed:
        for r in result.fusion.reasons:
            print(f"      · {r}")

    if show_heatmap and result.heatmap:
        cv2.imshow(f"Heatmap — {image_path.name}", result.heatmap.overlay)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        img_to_save = (result.heatmap.overlay
                       if result.heatmap else annotate_image(raw, result))
        out_name = f"{image_path.stem}_{result.verdict}{image_path.suffix}"
        save_image(img_to_save, save_dir / out_name)

    return {
        "file":              image_path.name,
        "verdict":           result.verdict,
        "latency_ms":        round(latency_ms, 1),
        "alignment_inliers": result.alignment_inliers,
        "roi_scores":        {k: round(v, 4) for k, v in result.roi_scores.items()},
        "lr_score":   round(result.lr_result.score, 4)       if result.lr_result      else None,
        "gm_score":   round(result.gm_result.similarity, 4)  if result.gm_result      else None,
        "barcode":    result.barcode_result.decoded_value     if result.barcode_result else None,
        "reasons":    result.fusion.reasons,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Himalaya label inspection.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image",  type=Path, help="Single image path")
    group.add_argument("--folder", type=Path, help="Folder of images")

    parser.add_argument("--show-heatmap", action="store_true",
                        help="Open heatmap window for each failed image")
    parser.add_argument("--save-output",  type=Path, default=None,
                        help="Save annotated images to this directory")
    parser.add_argument("--json-out",     type=Path, default=None,
                        help="Save all results as JSON to this file")
    args = parser.parse_args()

    print("\n[Loading pipeline...]")
    inspector = HimalayaLabelInspector.from_config(CONFIG_DIR)

    print("\n" + "=" * 60)
    print("  HIMALAYA LABEL INSPECTION")
    print("=" * 60 + "\n")

    records = []

    if args.image:
        rec = run_single(inspector, args.image,
                         args.show_heatmap, args.save_output)
        records.append(rec)

    else:
        image_paths = list_images(args.folder)
        if not image_paths:
            print(f"[ERROR] No images found in {args.folder}")
            sys.exit(1)

        print(f"Inspecting {len(image_paths)} images...\n")
        n_pass = n_fail = 0

        for img_path in image_paths:
            rec = run_single(inspector, img_path,
                             args.show_heatmap, args.save_output)
            records.append(rec)
            if rec["verdict"] == "PASS":
                n_pass += 1
            else:
                n_fail += 1

        avg_ms = (sum(r["latency_ms"] for r in records) / len(records)
                  if records else 0)

        print(f"\n── Summary ─────────────────────────────────────────")
        print(f"  Total:        {len(records)}")
        print(f"  PASS:         {n_pass}")
        print(f"  FAIL:         {n_fail}")
        print(f"  Avg latency:  {avg_ms:.1f} ms/image")

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.json_out, "w") as f:
            json.dump(records, f, indent=2)
        print(f"\n[OK] Results saved → {args.json_out}")


if __name__ == "__main__":
    main()
