"""
training/train_nsc_pipeline.py
================================
Unified orchestrator for the NSC cylindrical tube rotation-unwrap anomaly
detection training pipeline.

Runs all stages in order:
    Stage 0: Quality gate calibration (classical CV)
    Stage 1: Geometric registration calibration (classical CV)
    Data:    Patch extraction + golden profile construction
    Stage 3: PatchCore anomaly detection training (GPU)
    Final:   Threshold calibration + summary report

Usage:
    # Full pipeline (recommended for first run)
    python training/train_nsc_pipeline.py --data-root dataset/NSC --device cuda:0

    # Skip to PatchCore training (data already prepared)
    python training/train_nsc_pipeline.py --skip-prepare --skip-gate --skip-registration

    # Only prepare data (no GPU needed)
    python training/train_nsc_pipeline.py --data-root dataset/NSC --skip-patchcore

    # Smoke test (few images, small coreset)
    python training/train_nsc_pipeline.py --data-root dataset/NSC --max-images 5 --coreset-ratio 0.01

    # CPU-only (no GPU required, slower PatchCore)
    python training/train_nsc_pipeline.py --data-root dataset/NSC --device cpu --batch 8
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="NSC Rotation-Unwrap Anomaly Detection — Full Training Pipeline"
    )

    # Data paths
    parser.add_argument("--data-root", type=Path, default=Path("dataset/NSC"),
                        help="Raw dataset directory")
    parser.add_argument("--prepared-root", type=Path, default=Path("dataset/NSC_prepared"),
                        help="Output directory for prepared data")
    parser.add_argument("--config", type=Path,
                        default=Path("configs/sku_profiles/nsc_tube_unwrap.yaml"),
                        help="SKU profile YAML config")

    # Stage control
    parser.add_argument("--skip-gate", action="store_true",
                        help="Skip Stage 0 quality gate calibration")
    parser.add_argument("--skip-registration", action="store_true",
                        help="Skip Stage 1 registration calibration")
    parser.add_argument("--skip-prepare", action="store_true",
                        help="Skip data preparation (patches + golden profile)")
    parser.add_argument("--skip-patchcore", action="store_true",
                        help="Skip Stage 3b PatchCore training")

    # Training hyperparameters
    parser.add_argument("--batch", type=int, default=16,
                        help="Batch size for PatchCore (default: 16 for A500 8GB)")
    parser.add_argument("--coreset-ratio", type=float, default=0.10,
                        help="PatchCore coreset ratio (default: 0.10)")
    parser.add_argument("--k-nearest", type=int, default=9,
                        help="k-NN k for PatchCore scoring (default: 9)")
    parser.add_argument("--patch-size", type=int, default=384,
                        help="Patch size for extraction (default: 384)")
    parser.add_argument("--stride", type=int, default=192,
                        help="Patch stride (default: 192, = 50%% overlap)")
    parser.add_argument("--target-recall", type=float, default=0.99,
                        help="Target recall for threshold calibration (default: 0.99)")

    # Hardware
    parser.add_argument("--device", type=str, default="",
                        help="Device: 'cuda:0', 'cpu', '' for auto")
    parser.add_argument("--workers", type=int, default=4,
                        help="DataLoader workers (default: 4)")

    # Debugging
    parser.add_argument("--max-images", type=int, default=0,
                        help="Process only first N images per class (0=all)")
    parser.add_argument("--visualize", action="store_true",
                        help="Save visualizations for gate and registration")
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def run_stage(name: str, cmd: list[str], stage_num: str) -> bool:
    """Run a subprocess stage and return success/failure."""
    print(f"\n{'═' * 70}")
    print(f"  [{stage_num}] {name}")
    print(f"{'═' * 70}\n")

    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(Path.cwd()))
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"\n  ❌ {name} FAILED (exit code {result.returncode})")
        return False

    print(f"\n  ✓ {name} completed in {elapsed:.1f}s")
    return True


def main() -> None:
    args = parse_args()
    python = sys.executable

    print(f"\n{'═' * 70}")
    print("  NSC Rotation-Unwrap Anomaly Detection Pipeline")
    print(f"{'═' * 70}")
    print(f"  Data root     : {args.data_root}")
    print(f"  Prepared root : {args.prepared_root}")
    print(f"  Config        : {args.config}")
    print(f"  Device        : {args.device or 'auto'}")
    print(f"  Batch size    : {args.batch}")
    print(f"  Patch size    : {args.patch_size}")
    print(f"  Coreset ratio : {args.coreset_ratio}")
    print(f"  Max images    : {args.max_images or 'all'}")
    print(f"{'═' * 70}")

    stages_run = []
    stages_skipped = []
    stages_failed = []
    pipeline_start = time.time()

    # ─── Stage 0: Quality Gate Calibration ───────────────────────────────
    if not args.skip_gate:
        cmd = [
            python, "training/train_nsc_quality_gate.py",
            "--data-root", str(args.data_root),
            "--config", str(args.config),
            "--output", "runs/nsc_quality_gate",
        ]
        if args.max_images > 0:
            cmd += ["--max-images", str(args.max_images)]
        if args.visualize:
            cmd.append("--visualize")

        if run_stage("Quality Gate Calibration (Stage 0)", cmd, "0/4"):
            stages_run.append("quality_gate")
        else:
            stages_failed.append("quality_gate")
            print("\n  ⚠️  Quality gate failed but continuing (non-blocking)...")
    else:
        stages_skipped.append("quality_gate")
        print("\n  [SKIP] Stage 0: Quality Gate Calibration")

    # ─── Stage 1: Registration Calibration ───────────────────────────────
    if not args.skip_registration:
        cmd = [
            python, "training/train_nsc_registration.py",
            "--data-root", str(args.data_root),
            "--config", str(args.config),
            "--output", "runs/nsc_registration",
        ]
        if args.max_images > 0:
            cmd += ["--max-images", str(args.max_images)]
        if args.visualize:
            cmd.append("--visualize")

        if run_stage("Geometric Registration Calibration (Stage 1)", cmd, "1/4"):
            stages_run.append("registration")
        else:
            stages_failed.append("registration")
            print("\n  ⚠️  Registration failed but continuing (non-blocking)...")
    else:
        stages_skipped.append("registration")
        print("\n  [SKIP] Stage 1: Registration Calibration")

    # ─── Data Preparation (Patches + Golden Profile) ─────────────────────
    if not args.skip_prepare:
        cmd = [
            python, "scripts/prepare_nsc_dataset.py",
            "--data-root", str(args.data_root),
            "--output", str(args.prepared_root),
            "--config", str(args.config),
            "--patch-size", str(args.patch_size),
            "--stride", str(args.stride),
            "--seed", str(args.seed),
        ]
        if args.max_images > 0:
            cmd += ["--max-images", str(args.max_images)]

        if run_stage("Data Preparation (Patches + Golden Profile)", cmd, "2/4"):
            stages_run.append("data_preparation")
        else:
            stages_failed.append("data_preparation")
            if not args.skip_patchcore:
                print("\n  ❌ Data preparation failed — cannot proceed to PatchCore.")
                print("     Fix data preparation issues and re-run.")
                _print_summary(stages_run, stages_skipped, stages_failed,
                               time.time() - pipeline_start)
                return
    else:
        stages_skipped.append("data_preparation")
        print("\n  [SKIP] Data Preparation")

    # ─── Stage 3b: PatchCore Training ────────────────────────────────────
    if not args.skip_patchcore:
        cmd = [
            python, "training/train_nsc_patchcore.py",
            "--data-root", str(args.prepared_root),
            "--output", "runs/nsc_patchcore",
            "--batch", str(args.batch),
            "--coreset-ratio", str(args.coreset_ratio),
            "--k-nearest", str(args.k_nearest),
            "--image-size", str(args.patch_size),
            "--target-recall", str(args.target_recall),
            "--seed", str(args.seed),
            "--workers", str(args.workers),
        ]
        if args.device:
            cmd += ["--device", args.device]
        if args.max_images > 0:
            cmd += ["--max-patches", str(args.max_images * 50)]  # Rough estimate

        if run_stage("PatchCore Anomaly Detection Training (Stage 3b)", cmd, "3/4"):
            stages_run.append("patchcore")
        else:
            stages_failed.append("patchcore")
    else:
        stages_skipped.append("patchcore")
        print("\n  [SKIP] Stage 3b: PatchCore Training")

    # ─── Final Summary ───────────────────────────────────────────────────
    total_time = time.time() - pipeline_start
    _print_summary(stages_run, stages_skipped, stages_failed, total_time)

    # Generate final summary report
    _save_pipeline_report(args, stages_run, stages_skipped, stages_failed, total_time)


def _print_summary(
    stages_run: list[str],
    stages_skipped: list[str],
    stages_failed: list[str],
    total_time: float,
) -> None:
    print(f"\n{'═' * 70}")
    print("  Pipeline Summary")
    print(f"{'═' * 70}")
    print(f"  ✓ Completed : {', '.join(stages_run) or 'none'}")
    print(f"  ⏭ Skipped   : {', '.join(stages_skipped) or 'none'}")
    print(f"  ❌ Failed    : {', '.join(stages_failed) or 'none'}")
    print(f"  Total time  : {total_time / 60:.1f} minutes")

    if not stages_failed:
        print(f"\n  🎉 All requested stages completed successfully!")
    else:
        print(f"\n  ⚠️  {len(stages_failed)} stage(s) failed. Review logs above.")

    # Print expected outputs
    print(f"\n  Expected outputs:")
    if "quality_gate" in stages_run:
        print(f"    Quality gate  : runs/nsc_quality_gate/quality_gate_report.json")
    if "registration" in stages_run:
        print(f"    Registration  : runs/nsc_registration/registration_report.json")
    if "data_preparation" in stages_run:
        print(f"    Prepared data : dataset/NSC_prepared/")
        print(f"    Golden profile: dataset/NSC_prepared/golden_profile/")
    if "patchcore" in stages_run:
        print(f"    Memory bank   : models/nsc_patchcore_membank.pt")
        print(f"    PatchCore cfg : models/nsc_patchcore_config.yaml")
        print(f"    Backbone ONNX : models/nsc_patchcore_backbone.onnx")
        print(f"    Training rpt  : runs/nsc_patchcore/patchcore_training_report.json")

    print(f"{'═' * 70}\n")


def _save_pipeline_report(
    args: argparse.Namespace,
    stages_run: list[str],
    stages_skipped: list[str],
    stages_failed: list[str],
    total_time: float,
) -> None:
    """Save a JSON report summarizing the pipeline run."""
    report_dir = Path("runs/nsc_pipeline")
    report_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "pipeline": "nsc_rotation_unwrap_anomaly_detection",
        "stages_run": stages_run,
        "stages_skipped": stages_skipped,
        "stages_failed": stages_failed,
        "total_time_seconds": round(total_time, 1),
        "config": str(args.config),
        "data_root": str(args.data_root),
        "prepared_root": str(args.prepared_root),
        "hyperparameters": {
            "batch_size": args.batch,
            "patch_size": args.patch_size,
            "stride": args.stride,
            "coreset_ratio": args.coreset_ratio,
            "k_nearest": args.k_nearest,
            "target_recall": args.target_recall,
            "device": args.device or "auto",
            "seed": args.seed,
        },
    }

    report_path = report_dir / "pipeline_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
