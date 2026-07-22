"""
run_roi_inference.py
────────────────────
Person A — integration script (Day 3-4).

Runs the ROI-based inspection pipeline on a single image or folder of images.
Outputs:
  - Annotated PNG with bounding boxes + heatmap drawn on the label
  - JSON result with per-ROI scores, thresholds, and bounding box coordinates

Usage:
    # Single image
    python scripts/run_roi_inference.py --image "dataset/NSC/NSC BAD IMAGES/1.bmp"

    # Entire folder
    python scripts/run_roi_inference.py --folder "dataset/NSC/NSC BAD IMAGES"

    # With custom output directory
    python scripts/run_roi_inference.py --folder "dataset/NSC/NSC BAD IMAGES" \\
        --save-output outputs/results/ --json-out outputs/run_results.json

    # Show annotated image in a window (requires display)
    python scripts/run_roi_inference.py --image "..." --show
"""

from __future__ import annotations

import sys
import json
import argparse
import time
import cv2
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.roi_pipeline import ROIInspector
from src.utils.image_utils import list_images

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR   = PROJECT_ROOT / "config"
MODELS_ROOT  = PROJECT_ROOT / "models" / "rois"

SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


def run_single_image(
    inspector: ROIInspector,
    image_path: Path,
    save_dir: Path | None = None,
    show: bool = False,
) -> dict:
    """
    Inspect one image. Print result. Optionally save annotated image.

    Returns the JSON-serialisable result dict.
    """
    # Load full NSC image as BGR color (full images are RGB BMP)
    # The ROI pipeline converts to grayscale internally for model scoring
    raw = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if raw is None:
        print(f"  [ERROR] Could not read: {image_path}")
        return {"file": image_path.name, "verdict": "ERROR", "error": "imread failed"}

    t0     = time.perf_counter()
    result = inspector.inspect(raw, image_path=str(image_path))
    elapsed_ms = (time.perf_counter() - t0) * 1000

    # Print summary line
    icon = "✓" if result.overall_pass else "✗"
    print(f"  {icon}  {image_path.name:<40}  "
          f"{result.overall_verdict}  {elapsed_ms:6.1f} ms")

    if not result.overall_pass:
        for reason in result.fail_reasons:
            print(f"       → {reason}")

    # Show in window
    if show and result.annotated_image is not None:
        win_name = f"NSC Inspection — {image_path.name}"
        # Resize for display if image is very tall
        disp = result.annotated_image
        max_h = 900
        if disp.shape[0] > max_h:
            scale = max_h / disp.shape[0]
            disp = cv2.resize(disp, (int(disp.shape[1] * scale), max_h))
        cv2.imshow(win_name, disp)
        cv2.waitKey(0)
        cv2.destroyWindow(win_name)

    # Save annotated image
    if save_dir is not None and result.annotated_image is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        out_stem = f"{image_path.stem}_{result.overall_verdict}"
        out_path = save_dir / f"{out_stem}.png"
        cv2.imwrite(str(out_path), result.annotated_image)

    return result.to_dict()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run per-ROI Himalaya NSC label inspection."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image",  type=Path, help="Single image path")
    group.add_argument("--folder", type=Path, help="Folder of images to inspect")

    parser.add_argument(
        "--save-output", type=Path, default=None, metavar="DIR",
        help="Save annotated output images to this directory"
    )
    parser.add_argument(
        "--json-out", type=Path, default=None, metavar="FILE",
        help="Save all results as a JSON file"
    )
    parser.add_argument(
        "--show", action="store_true",
        help="Display each annotated image in a window (press any key to continue)"
    )
    parser.add_argument(
        "--config", type=Path, default=CONFIG_DIR,
        help=f"Config directory (default: {CONFIG_DIR})"
    )
    parser.add_argument(
        "--models", type=Path, default=MODELS_ROOT,
        help=f"Trained ROI models directory (default: {MODELS_ROOT})"
    )
    args = parser.parse_args()

    # ── Load pipeline ─────────────────────────────────────────────────────────
    print("\n[Loading ROI Inspector pipeline...]")
    try:
        inspector = ROIInspector.from_config(
            config_dir=args.config,
            models_root=args.models,
        )
    except Exception as exc:
        print(f"[ERROR] Failed to load inspector: {exc}")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("  HIMALAYA NSC — ROI INSPECTION")
    print("=" * 60 + "\n")

    records = []

    # ── Single image mode ─────────────────────────────────────────────────────
    if args.image:
        if not args.image.exists():
            print(f"[ERROR] File not found: {args.image}")
            sys.exit(1)
        rec = run_single_image(inspector, args.image, args.save_output, args.show)
        records.append(rec)

    # ── Folder mode ───────────────────────────────────────────────────────────
    else:
        if not args.folder.exists():
            print(f"[ERROR] Folder not found: {args.folder}")
            sys.exit(1)

        image_paths = list_images(args.folder)
        if not image_paths:
            print(f"[ERROR] No images found in {args.folder}")
            sys.exit(1)

        print(f"Inspecting {len(image_paths)} images in {args.folder}\n")

        n_pass = n_fail = n_err = 0
        latencies = []

        for img_path in image_paths:
            rec = run_single_image(inspector, img_path, args.save_output, args.show)
            records.append(rec)

            if rec.get("overall_verdict") == "PASS":
                n_pass += 1
            elif rec.get("overall_verdict") == "FAIL":
                n_fail += 1
            else:
                n_err += 1

            if "latency_ms" in rec:
                latencies.append(rec["latency_ms"])

        avg_ms = sum(latencies) / len(latencies) if latencies else 0.0

        print(f"\n{'─' * 60}")
        print(f"  SUMMARY")
        print(f"{'─' * 60}")
        print(f"  Total:         {len(records)}")
        print(f"  PASS:          {n_pass}")
        print(f"  FAIL:          {n_fail}")
        if n_err > 0:
            print(f"  ERROR:         {n_err}")
        print(f"  Avg latency:   {avg_ms:.1f} ms/image")
        print(f"  False positive rate would need ground truth to compute.")

    # ── Save JSON ─────────────────────────────────────────────────────────────
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.json_out, "w") as f:
            json.dump(records, f, indent=2, default=str)
        print(f"\n[OK] Results saved → {args.json_out}")

    if args.save_output:
        print(f"[OK] Annotated images saved → {args.save_output}")


if __name__ == "__main__":
    main()
