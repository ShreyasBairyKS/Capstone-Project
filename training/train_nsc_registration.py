"""
training/train_nsc_registration.py
====================================
Stage 1 — Build and validate the geometric registration pipeline for NSC
cylindrical tube rotation-unwrap images.

This script:
  1. Extracts a fiducial template from a reference good image (or uses an existing one)
  2. Runs template matching on all images to find per-image rotation offsets
  3. Validates 2-fold rotational symmetry via cross-correlation on multiple bands
  4. Flags images with significant offset deviation as potential print-registration defects
  5. Outputs registration config, per-image offset log, and optional visualizations

Usage:
    # Full registration calibration
    python training/train_nsc_registration.py

    # With visualizations
    python training/train_nsc_registration.py --visualize

    # Use specific reference image for template extraction
    python training/train_nsc_registration.py --reference-image "dataset/NSC/NSC GOOD IMAGES/1.bmp"

    # Custom config
    python training/train_nsc_registration.py --config configs/sku_profiles/nsc_tube_unwrap.yaml
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.prepare_nsc_dataset import (
    extract_fiducial_template,
    find_fiducial_offset,
    measure_symmetry_offset,
    register_strip,
    load_config,
    list_images,
    load_image,
)


# ─────────────────────────────────────────────────────────────────────────────
# Zone Segmentation Validation
# ─────────────────────────────────────────────────────────────────────────────

def validate_zone_segmentation(
    image_bgr: np.ndarray,
    zone_rows: dict,
) -> dict:
    """
    Validate that zone segmentation row ranges produce sensible regions.

    Checks:
    - Cap/crimp band: should have distinct horizontal edge features (the crimp seam)
    - Label band: should have the highest contrast / information content
    - Seal/bottom band: typically lower contrast

    Returns dict with per-zone statistics.
    """
    stats = {}
    for zone_name, (y_start, y_end) in zone_rows.items():
        zone = image_bgr[y_start:y_end, :, :]
        gray = cv2.cvtColor(zone, cv2.COLOR_BGR2GRAY)

        stats[zone_name] = {
            "rows": [y_start, y_end],
            "height": y_end - y_start,
            "mean_intensity": float(np.mean(gray)),
            "std_intensity": float(np.std(gray)),
            "laplacian_var": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
            "edge_density": float(np.mean(cv2.Canny(gray, 50, 150) > 0)),
        }

    return stats


def compute_self_similarity_score(
    image_bgr: np.ndarray,
    zone_rows: dict,
    offset: int,
) -> dict:
    """
    Compute self-similarity between the two half-repeats of the label band.

    Returns correlation and difference statistics.
    """
    label_start, label_end = zone_rows["label_band"]
    label_band = cv2.cvtColor(
        image_bgr[label_start:label_end, :, :], cv2.COLOR_BGR2GRAY
    ).astype(np.float64)

    w = label_band.shape[1]

    # Shift by measured offset
    shifted = np.roll(label_band, -offset, axis=1)

    # Compare overlapping region (exclude wrap-around edges)
    margin = 100
    region_orig = label_band[:, margin : w - margin]
    region_shifted = shifted[:, margin : w - margin]

    # Pixel-wise absolute difference
    diff = np.abs(region_orig - region_shifted)

    # Correlation coefficient
    mean_o = np.mean(region_orig)
    mean_s = np.mean(region_shifted)
    std_o = np.std(region_orig)
    std_s = np.std(region_shifted)

    if std_o > 0 and std_s > 0:
        corr = float(np.mean(
            (region_orig - mean_o) * (region_shifted - mean_s)
        ) / (std_o * std_s))
    else:
        corr = 0.0

    return {
        "correlation": round(corr, 4),
        "mean_diff": round(float(np.mean(diff)), 2),
        "median_diff": round(float(np.median(diff)), 2),
        "max_diff": round(float(np.max(diff)), 2),
        "diff_above_30": round(float(np.mean(diff > 30)), 4),  # Fraction of high-diff pixels
    }


# ─────────────────────────────────────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────────────────────────────────────

def save_registration_visualization(
    image_bgr: np.ndarray,
    registered: np.ndarray,
    zone_rows: dict,
    fiducial_offset: int,
    symmetry_offset: int,
    img_name: str,
    output_dir: Path,
) -> None:
    """
    Save a visualization showing:
    - Original image with fiducial location marked
    - Registered image with zone boundaries
    - Self-similarity diff overlay
    """
    scale = 0.2
    h, w = image_bgr.shape[:2]

    # Registered image with zone lines
    viz = cv2.resize(registered, None, fx=scale, fy=scale)
    sh, sw = viz.shape[:2]

    for zone_name, (y_start, y_end) in zone_rows.items():
        y_scaled = int(y_start * scale)
        cv2.line(viz, (0, y_scaled), (sw, y_scaled), (0, 255, 0), 2)
        cv2.putText(viz, zone_name, (10, y_scaled + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Mark symmetry offset
    x_sym = int(symmetry_offset * scale)
    cv2.line(viz, (x_sym, 0), (x_sym, sh), (0, 0, 255), 2)
    cv2.putText(viz, f"sym={symmetry_offset}", (x_sym + 5, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    output_path = output_dir / f"reg_viz_{img_name}.jpg"
    cv2.imwrite(str(output_path), viz, [cv2.IMWRITE_JPEG_QUALITY, 85])


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and validate NSC geometric registration (Stage 1)"
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
        "--output", type=Path, default=Path("runs/nsc_registration"),
        help="Output directory for registration results",
    )
    parser.add_argument(
        "--reference-image", type=Path, default=None,
        help="Specific good image to use for fiducial template extraction",
    )
    parser.add_argument("--visualize", action="store_true",
                        help="Save registration visualization images")
    parser.add_argument("--max-images", type=int, default=0,
                        help="Process only first N images per class (0=all)")
    parser.add_argument("--offset-tolerance", type=int, default=None,
                        help="Override offset deviation tolerance (px)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    data_root = args.data_root.resolve()
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    reg_cfg = config["registration"]
    zone_rows = reg_cfg["zone_rows"]
    expected_offset = reg_cfg["expected_symmetry_offset"]
    offset_tol = args.offset_tolerance or reg_cfg.get("offset_tolerance", 50)

    good_dir = data_root / "NSC GOOD IMAGES"
    bad_dir = data_root / "NSC BAD IMAGES"

    good_images = list_images(good_dir)
    bad_images = list_images(bad_dir)

    if args.max_images > 0:
        good_images = good_images[: args.max_images]
        bad_images = bad_images[: args.max_images]

    print(f"\n{'=' * 70}")
    print("  NSC Geometric Registration Calibration (Stage 1)")
    print(f"{'=' * 70}")
    print(f"  Good images      : {len(good_images)}")
    print(f"  Bad images       : {len(bad_images)}")
    print(f"  Expected offset  : {expected_offset} px")
    print(f"  Offset tolerance : ±{offset_tol} px")
    print(f"{'=' * 70}")

    # ── Step 1: Extract or load fiducial template ──
    print("\n[STEP 1] Fiducial template...")
    template_path = Path(reg_cfg.get("fiducial_template",
                                     "configs/templates/nsc_logo_template.png"))

    if template_path.exists():
        print(f"  Using existing template: {template_path}")
        fiducial_template = cv2.imread(str(template_path), cv2.IMREAD_COLOR)
        if fiducial_template is None:
            raise IOError(f"Failed to load template: {template_path}")
    else:
        ref_path = args.reference_image or good_images[0]
        print(f"  Extracting template from: {ref_path}")
        ref_img = load_image(ref_path)
        fiducial_template = extract_fiducial_template(
            ref_img, zone_rows, save_path=template_path
        )
        del ref_img

    print(f"  Template size: {fiducial_template.shape[1]}×{fiducial_template.shape[0]}")

    # ── Step 2: Process all images ──
    all_images = [(p, "good") for p in good_images] + [(p, "bad") for p in bad_images]

    registration_log = []
    zone_stats_log = []
    similarity_log = []
    flagged_drift = []

    viz_dir = output_dir / "visualizations"
    if args.visualize:
        viz_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[STEP 2] Processing {len(all_images)} images...")

    for i, (img_path, label) in enumerate(all_images):
        t0 = time.time()

        img = load_image(img_path)

        # Template matching for fiducial offset
        x_offset, match_score = find_fiducial_offset(
            img, fiducial_template, zone_rows,
            threshold=reg_cfg.get("template_match_threshold", 0.4),
        )

        # Symmetry offset measurement
        sym_offset, sym_conf, sym_details = measure_symmetry_offset(
            img, zone_rows,
            expected_offset=expected_offset,
            num_bands=reg_cfg.get("num_correlation_bands", 10),
            band_height=reg_cfg.get("correlation_band_height", 100),
        )

        # Offset deviation check
        deviation = abs(sym_offset - expected_offset)
        is_drift = deviation > offset_tol

        if is_drift:
            flagged_drift.append({
                "image": img_path.name,
                "label": label,
                "symmetry_offset": sym_offset,
                "deviation": deviation,
            })

        # Register
        registered = register_strip(img, x_offset, image_width=img.shape[1])

        # Zone validation
        zone_stats = validate_zone_segmentation(registered, zone_rows)
        zone_stats_log.append({
            "image": img_path.name,
            "label": label,
            "zones": zone_stats,
        })

        # Self-similarity score
        sim_result = compute_self_similarity_score(registered, zone_rows, sym_offset)
        similarity_log.append({
            "image": img_path.name,
            "label": label,
            **sim_result,
        })

        # Registration log entry
        entry = {
            "image": img_path.name,
            "label": label,
            "fiducial_x_offset": x_offset,
            "fiducial_match_score": round(match_score, 4),
            "symmetry_offset": sym_offset,
            "symmetry_confidence": round(sym_conf, 4),
            "offset_deviation": deviation,
            "is_drift": is_drift,
            "self_similarity_corr": sim_result["correlation"],
        }
        registration_log.append(entry)

        # Visualization
        if args.visualize:
            save_registration_visualization(
                img, registered, zone_rows,
                x_offset, sym_offset, img_path.stem, viz_dir
            )

        elapsed = time.time() - t0
        status = "DRIFT!" if is_drift else "OK"
        print(
            f"  [{i + 1}/{len(all_images)}] {img_path.name} ({label}): "
            f"fid_off={x_offset} match={match_score:.3f} "
            f"sym_off={sym_offset} conf={sym_conf:.2f} "
            f"corr={sim_result['correlation']:.3f} "
            f"[{status}] ({elapsed:.1f}s)"
        )

        del img, registered

    # ── Step 3: Statistical summary ──
    print(f"\n[STEP 3] Registration Statistics")

    offsets = [e["fiducial_x_offset"] for e in registration_log]
    match_scores = [e["fiducial_match_score"] for e in registration_log]
    sym_offsets = [e["symmetry_offset"] for e in registration_log]
    sym_confs = [e["symmetry_confidence"] for e in registration_log]
    corrs_good = [e["self_similarity_corr"] for e in registration_log if e["label"] == "good"]
    corrs_bad = [e["self_similarity_corr"] for e in registration_log if e["label"] == "bad"]

    print(f"\n  Fiducial X-offset: mean={np.mean(offsets):.1f}, "
          f"std={np.std(offsets):.1f}, range=[{min(offsets)}, {max(offsets)}]")
    print(f"  Match score: mean={np.mean(match_scores):.3f}, "
          f"min={min(match_scores):.3f}, max={max(match_scores):.3f}")
    print(f"  Symmetry offset: mean={np.mean(sym_offsets):.1f}, "
          f"std={np.std(sym_offsets):.1f}, range=[{min(sym_offsets)}, {max(sym_offsets)}]")
    print(f"  Symmetry confidence: mean={np.mean(sym_confs):.3f}")

    if corrs_good:
        print(f"  Self-similarity (GOOD): mean={np.mean(corrs_good):.3f}, "
              f"min={min(corrs_good):.3f}")
    if corrs_bad:
        print(f"  Self-similarity (BAD) : mean={np.mean(corrs_bad):.3f}, "
              f"min={min(corrs_bad):.3f}")

    if flagged_drift:
        print(f"\n  [WARN]  {len(flagged_drift)} images flagged for print registration drift:")
        for f in flagged_drift:
            print(f"    - {f['image']} ({f['label']}): offset={f['symmetry_offset']}, "
                  f"deviation={f['deviation']}px")
    else:
        print(f"\n  [OK] No print registration drift detected (tolerance: +/-{offset_tol}px)")

    # -- Step 4: Save results --
    print("\n[STEP 4] Saving results...")

    report = {
        "config": {
            "expected_symmetry_offset": expected_offset,
            "offset_tolerance": offset_tol,
            "template_path": str(template_path),
            "template_size": list(fiducial_template.shape[:2]),
        },
        "summary": {
            "total_images": len(all_images),
            "n_good": len(good_images),
            "n_bad": len(bad_images),
            "drift_flagged": len(flagged_drift),
            "fiducial_offset_mean": round(float(np.mean(offsets)), 1),
            "fiducial_offset_std": round(float(np.std(offsets)), 1),
            "symmetry_offset_mean": round(float(np.mean(sym_offsets)), 1),
            "match_score_mean": round(float(np.mean(match_scores)), 3),
            "self_sim_corr_good_mean": round(float(np.mean(corrs_good)), 3) if corrs_good else None,
            "self_sim_corr_bad_mean": round(float(np.mean(corrs_bad)), 3) if corrs_bad else None,
        },
        "per_image": registration_log,
        "zone_stats": zone_stats_log,
        "self_similarity": similarity_log,
        "drift_flagged": flagged_drift,
    }

    report_path = output_dir / "registration_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Save calibrated registration config
    calibrated = {
        "registration": {
            "expected_symmetry_offset": expected_offset,
            "measured_mean_offset": round(float(np.mean(sym_offsets)), 1),
            "offset_tolerance": offset_tol,
            "fiducial_template": str(template_path),
            "template_match_threshold": reg_cfg.get("template_match_threshold", 0.4),
            "zone_rows": zone_rows,
            "num_correlation_bands": reg_cfg.get("num_correlation_bands", 10),
        }
    }
    calibrated_path = output_dir / "calibrated_registration.yaml"
    with open(calibrated_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(calibrated, f, sort_keys=False)

    print(f"\n  Report -> {report_path}")
    print(f"  Config -> {calibrated_path}")
    if args.visualize:
        print(f"  Visualizations -> {viz_dir}")

    print(f"\n{'=' * 70}")
    print("  [OK] Registration calibration complete")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
