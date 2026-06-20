"""
training/train_nsc_quality_gate.py
===================================
Stage 0 — Calibrate and validate the capture quality gate for NSC cylindrical
tube rotation-unwrap images.

This is NOT a neural network training script. It builds and calibrates the
deterministic quality gate (skin-tone occlusion + specular glare detection)
and validates it across the full dataset.

The quality gate produces a THREE-WAY output:
  - PASS      → image is clean, proceed to anomaly detection
  - RECAPTURE → image is corrupted by occlusion/glare, re-capture required
  - (REJECT is NOT produced here — that's a downstream anomaly detection verdict)

Usage:
    # Run calibration on full dataset
    python training/train_nsc_quality_gate.py

    # Custom thresholds
    python training/train_nsc_quality_gate.py --skin-threshold 0.03 --glare-threshold 0.01

    # Save visualizations
    python training/train_nsc_quality_gate.py --visualize

    # Use specific config
    python training/train_nsc_quality_gate.py --config configs/sku_profiles/nsc_tube_unwrap.yaml
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

# Import quality gate functions from the preparation script
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.prepare_nsc_dataset import (
    detect_skin_occlusion,
    detect_glare,
    load_config,
    list_images,
    load_image,
)


# ─────────────────────────────────────────────────────────────────────────────
# Sweep & Calibration
# ─────────────────────────────────────────────────────────────────────────────

def sweep_skin_thresholds(
    images: list[tuple[Path, str]],
    config: dict,
    thresholds: list[float] | None = None,
) -> list[dict]:
    """
    Sweep occlusion coverage thresholds and report per-threshold statistics.

    Args:
        images: list of (path, label) where label is 'good' or 'bad'
        config: SKU profile config
        thresholds: list of threshold values to test

    Returns:
        List of dicts with threshold, n_flagged, flagged_good, flagged_bad
    """
    if thresholds is None:
        thresholds = [0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20]

    qg = config["quality_gate"]
    results = []

    # Pre-compute skin coverage for all images
    print("\n  Computing skin coverage for all images...")
    coverages = []
    for i, (img_path, label) in enumerate(images):
        img = load_image(img_path)
        coverage, _ = detect_skin_occlusion(
            img,
            hsv_lower=qg["skin_hsv_lower"],
            hsv_upper=qg["skin_hsv_upper"],
            hsv_lower_alt=qg.get("skin_hsv_lower_alt"),
            hsv_upper_alt=qg.get("skin_hsv_upper_alt"),
            min_area=qg.get("occlusion_min_area", 5000),
        )
        coverages.append((img_path.name, label, coverage))
        del img

        if (i + 1) % 10 == 0:
            print(f"    [{i + 1}/{len(images)}] processed")

    # Sweep thresholds
    for thresh in thresholds:
        flagged = [(name, label, cov) for name, label, cov in coverages if cov > thresh]
        flagged_good = sum(1 for _, l, _ in flagged if l == "good")
        flagged_bad = sum(1 for _, l, _ in flagged if l == "bad")
        n_good = sum(1 for _, l, _ in coverages if l == "good")
        n_bad = sum(1 for _, l, _ in coverages if l == "bad")

        results.append({
            "threshold": thresh,
            "n_flagged": len(flagged),
            "flagged_good": flagged_good,
            "flagged_bad": flagged_bad,
            "good_flag_rate": flagged_good / max(n_good, 1),
            "bad_flag_rate": flagged_bad / max(n_bad, 1),
        })

    return results


def sweep_glare_thresholds(
    images: list[tuple[Path, str]],
    config: dict,
    thresholds: list[float] | None = None,
) -> list[dict]:
    """Sweep glare area thresholds."""
    if thresholds is None:
        thresholds = [0.005, 0.01, 0.02, 0.03, 0.05, 0.10]

    qg = config["quality_gate"]
    results = []

    print("\n  Computing glare coverage for all images...")
    coverages = []
    critical_rows = qg.get("glare_critical_zone_rows")
    critical_zone = tuple(critical_rows) if critical_rows else None

    for i, (img_path, label) in enumerate(images):
        img = load_image(img_path)
        coverage, _ = detect_glare(
            img,
            value_threshold=qg["glare_value_threshold"],
            saturation_max=qg.get("glare_saturation_max", 30),
            critical_zone_rows=critical_zone,
        )
        coverages.append((img_path.name, label, coverage))
        del img

        if (i + 1) % 10 == 0:
            print(f"    [{i + 1}/{len(images)}] processed")

    for thresh in thresholds:
        flagged = [(name, label, cov) for name, label, cov in coverages if cov > thresh]
        flagged_good = sum(1 for _, l, _ in flagged if l == "good")
        flagged_bad = sum(1 for _, l, _ in flagged if l == "bad")
        n_good = sum(1 for _, l, _ in coverages if l == "good")
        n_bad = sum(1 for _, l, _ in coverages if l == "bad")

        results.append({
            "threshold": thresh,
            "n_flagged": len(flagged),
            "flagged_good": flagged_good,
            "flagged_bad": flagged_bad,
            "good_flag_rate": flagged_good / max(n_good, 1),
            "bad_flag_rate": flagged_bad / max(n_bad, 1),
        })

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────────────────────────────────────

def save_gate_visualization(
    img_path: Path,
    image_bgr: np.ndarray,
    skin_mask: np.ndarray,
    glare_mask: np.ndarray,
    output_dir: Path,
) -> None:
    """Save a side-by-side visualization of quality gate detections."""
    h, w = image_bgr.shape[:2]

    # Downsample for visualization (original is 1504×8000)
    scale = 0.25
    small_img = cv2.resize(image_bgr, None, fx=scale, fy=scale)
    small_skin = cv2.resize(skin_mask, None, fx=scale, fy=scale)
    small_glare = cv2.resize(glare_mask, None, fx=scale, fy=scale)

    # Overlay masks on image
    overlay = small_img.copy()
    overlay[small_skin > 0] = [0, 0, 255]   # Red for skin
    overlay[small_glare > 0] = [255, 255, 0]  # Cyan for glare

    blended = cv2.addWeighted(small_img, 0.6, overlay, 0.4, 0)

    output_path = output_dir / f"gate_viz_{img_path.stem}.jpg"
    cv2.imwrite(str(output_path), blended, [cv2.IMWRITE_JPEG_QUALITY, 85])


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate and validate NSC capture quality gate (Stage 0)"
    )
    parser.add_argument(
        "--data-root", type=Path, default=Path("dataset/NSC"),
        help="Root directory with 'NSC GOOD IMAGES' and 'NSC BAD IMAGES'",
    )
    parser.add_argument(
        "--config", type=Path, default=Path("configs/sku_profiles/nsc_tube_unwrap.yaml"),
        help="SKU profile YAML config",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("runs/nsc_quality_gate"),
        help="Output directory for calibration results",
    )
    parser.add_argument("--skin-threshold", type=float, default=None,
                        help="Override skin occlusion threshold")
    parser.add_argument("--glare-threshold", type=float, default=None,
                        help="Override glare area threshold")
    parser.add_argument("--visualize", action="store_true",
                        help="Save gate visualization images")
    parser.add_argument("--max-images", type=int, default=0,
                        help="Process only first N images per class (0=all)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    data_root = args.data_root.resolve()
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    good_dir = data_root / "NSC GOOD IMAGES"
    bad_dir = data_root / "NSC BAD IMAGES"

    good_images = list_images(good_dir)
    bad_images = list_images(bad_dir)

    if args.max_images > 0:
        good_images = good_images[: args.max_images]
        bad_images = bad_images[: args.max_images]

    all_images: list[tuple[Path, str]] = (
        [(p, "good") for p in good_images] +
        [(p, "bad") for p in bad_images]
    )

    print(f"\n{'=' * 70}")
    print("  NSC Quality Gate Calibration (Stage 0)")
    print(f"{'=' * 70}")
    print(f"  Good images: {len(good_images)}")
    print(f"  Bad images : {len(bad_images)}")
    print(f"  Config     : {args.config}")
    print(f"  Output     : {output_dir}")
    print(f"{'=' * 70}")

    # ── Threshold sweep: skin occlusion ──
    print("\n[1/4] Sweeping skin occlusion thresholds...")
    skin_results = sweep_skin_thresholds(all_images, config)
    print("\n  Skin Occlusion Threshold Sweep:")
    print(f"  {'Threshold':>10} {'Flagged':>8} {'Good':>6} {'Bad':>6} {'Good%':>8} {'Bad%':>8}")
    for r in skin_results:
        print(f"  {r['threshold']:>10.3f} {r['n_flagged']:>8} {r['flagged_good']:>6} "
              f"{r['flagged_bad']:>6} {r['good_flag_rate']:>7.1%} {r['bad_flag_rate']:>7.1%}")

    # ── Threshold sweep: glare ──
    print("\n[2/4] Sweeping glare thresholds...")
    glare_results = sweep_glare_thresholds(all_images, config)
    print("\n  Glare Threshold Sweep:")
    print(f"  {'Threshold':>10} {'Flagged':>8} {'Good':>6} {'Bad':>6} {'Good%':>8} {'Bad%':>8}")
    for r in glare_results:
        print(f"  {r['threshold']:>10.3f} {r['n_flagged']:>8} {r['flagged_good']:>6} "
              f"{r['flagged_bad']:>6} {r['good_flag_rate']:>7.1%} {r['bad_flag_rate']:>7.1%}")

    # ── Per-image quality gate with final thresholds ──
    qg = config["quality_gate"]
    skin_thresh = args.skin_threshold if args.skin_threshold is not None else qg["occlusion_threshold"]
    glare_thresh = args.glare_threshold if args.glare_threshold is not None else qg["glare_area_threshold"]

    print(f"\n[3/4] Running quality gate with final thresholds "
          f"(skin={skin_thresh}, glare={glare_thresh})...")

    # Temporarily override config thresholds if CLI args provided
    if args.skin_threshold is not None:
        config["quality_gate"]["occlusion_threshold"] = args.skin_threshold
    if args.glare_threshold is not None:
        config["quality_gate"]["glare_area_threshold"] = args.glare_threshold

    gate_results = []
    viz_dir = output_dir / "visualizations"
    if args.visualize:
        viz_dir.mkdir(parents=True, exist_ok=True)

    critical_rows = qg.get("glare_critical_zone_rows")
    critical_zone = tuple(critical_rows) if critical_rows else None

    for i, (img_path, label) in enumerate(all_images):
        img = load_image(img_path)

        # Skin detection
        skin_cov, skin_mask = detect_skin_occlusion(
            img,
            hsv_lower=qg["skin_hsv_lower"],
            hsv_upper=qg["skin_hsv_upper"],
            hsv_lower_alt=qg.get("skin_hsv_lower_alt"),
            hsv_upper_alt=qg.get("skin_hsv_upper_alt"),
            min_area=qg.get("occlusion_min_area", 5000),
        )

        # Glare detection
        glare_cov, glare_mask = detect_glare(
            img,
            value_threshold=qg["glare_value_threshold"],
            saturation_max=qg.get("glare_saturation_max", 30),
            critical_zone_rows=critical_zone,
        )

        # Determine status
        if skin_cov > skin_thresh:
            status = "RECAPTURE"
            reason = f"skin_occlusion ({skin_cov:.3f} > {skin_thresh})"
        elif glare_cov > glare_thresh:
            status = "RECAPTURE"
            reason = f"glare ({glare_cov:.3f} > {glare_thresh})"
        else:
            status = "PASS"
            reason = "clean"

        gate_results.append({
            "image": img_path.name,
            "label": label,
            "status": status,
            "reason": reason,
            "skin_coverage": round(skin_cov, 5),
            "glare_coverage": round(glare_cov, 5),
        })

        if args.visualize and status == "RECAPTURE":
            save_gate_visualization(img_path, img, skin_mask, glare_mask, viz_dir)

        del img

    # ── Summary ──
    print(f"\n[4/4] Quality Gate Summary:")
    pass_good = sum(1 for r in gate_results if r["label"] == "good" and r["status"] == "PASS")
    pass_bad = sum(1 for r in gate_results if r["label"] == "bad" and r["status"] == "PASS")
    recap_good = sum(1 for r in gate_results if r["label"] == "good" and r["status"] == "RECAPTURE")
    recap_bad = sum(1 for r in gate_results if r["label"] == "bad" and r["status"] == "RECAPTURE")

    print(f"\n  {'':>15} {'PASS':>8} {'RECAPTURE':>12}")
    print(f"  {'Good images':>15} {pass_good:>8} {recap_good:>12}")
    print(f"  {'Bad images':>15} {pass_bad:>8} {recap_bad:>12}")

    # Flag any good images that got RECAPTURE'd — these need manual review
    recaptured_good = [r for r in gate_results if r["label"] == "good" and r["status"] == "RECAPTURE"]
    if recaptured_good:
        print(f"\n  ⚠️  {len(recaptured_good)} GOOD images flagged for RECAPTURE:")
        for r in recaptured_good:
            print(f"    - {r['image']}: {r['reason']}")

    # Save results
    report = {
        "config": {
            "skin_threshold": skin_thresh,
            "glare_threshold": glare_thresh,
        },
        "summary": {
            "total_images": len(all_images),
            "good_pass": pass_good,
            "good_recapture": recap_good,
            "bad_pass": pass_bad,
            "bad_recapture": recap_bad,
        },
        "skin_sweep": skin_results,
        "glare_sweep": glare_results,
        "per_image": gate_results,
    }

    report_path = output_dir / "quality_gate_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Save calibrated config
    calibrated_config = {
        "quality_gate": {
            "skin_hsv_lower": qg["skin_hsv_lower"],
            "skin_hsv_upper": qg["skin_hsv_upper"],
            "skin_hsv_lower_alt": qg.get("skin_hsv_lower_alt"),
            "skin_hsv_upper_alt": qg.get("skin_hsv_upper_alt"),
            "occlusion_threshold": skin_thresh,
            "occlusion_min_area": qg.get("occlusion_min_area", 5000),
            "glare_value_threshold": qg["glare_value_threshold"],
            "glare_saturation_max": qg.get("glare_saturation_max", 30),
            "glare_area_threshold": glare_thresh,
            "glare_critical_zone_rows": qg.get("glare_critical_zone_rows"),
        }
    }
    calibrated_path = output_dir / "calibrated_quality_gate.yaml"
    with open(calibrated_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(calibrated_config, f, sort_keys=False)

    print(f"\n  Report → {report_path}")
    print(f"  Config → {calibrated_path}")
    if args.visualize:
        print(f"  Visualizations → {viz_dir}")
    print(f"\n{'=' * 70}")
    print("  ✓ Quality gate calibration complete")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
