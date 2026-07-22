"""
train_all_rois.py
─────────────────
Person B — Day 1/2 script.

Trains one EfficientAD anomaly model per ROI folder found under the
rois_root directory. The 4 pre-cropped grayscale ROI folders are expected
to already be present on the remote PC.

Expected input layout:
    <rois_root>/
        <any_folder_name>/
            good/      ← grayscale good crops (training data)
            bad/       ← grayscale bad crops  (threshold calibration, not used here)
        <any_folder_name>/
            good/
            bad/
        ...

Output layout:
    <models_root>/
        <roi_folder_name>/     ← one dir per ROI
            model_meta.json
            <anomalib checkpoint files>
        roi_model_paths.json   ← maps roi_name → checkpoint path (used by inference)

Usage:
    # Train all ROIs with EfficientAD (recommended):
    python scripts/train_all_rois.py

    # Use PatchCore instead (faster "training", higher RAM):
    python scripts/train_all_rois.py --model patchcore

    # Specify custom data and output directories:
    python scripts/train_all_rois.py --rois-root /path/to/rois --models-root /path/to/models

    # Train only specific ROI folders:
    python scripts/train_all_rois.py --only ROI_LOGO ROI_BARCODE
"""

import sys
import json
import argparse
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.training.train_roi_efficientad import train_roi_model, verify_roi_folder

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ROIS_ROOT   = PROJECT_ROOT / "data" / "rois"
DEFAULT_MODELS_ROOT = PROJECT_ROOT / "models" / "rois"

# Training hyperparameters — tuned for grayscale label crops on A500 GPU
TRAIN_CONFIG = {
    "image_size":         (256, 256),
    "max_epochs":         100,           # EfficientAD: needs ~100 epochs to converge
    "train_batch_size":   8,             # lower if GPU OOM; A500 should handle 16
    "num_workers":        4,             # set to 0 on Windows if DataLoader errors
    "model_size":         "small",       # EfficientAD: "small" (faster) or "medium"
    # PatchCore-specific (only used if --model patchcore)
    "coreset_sampling_ratio": 0.1,
    "num_neighbors":          9,
}


def discover_roi_folders(rois_root: Path) -> list[Path]:
    """Find all subdirectories with a 'good/' subfolder."""
    return sorted([
        d for d in rois_root.iterdir()
        if d.is_dir() and (d / "good").exists()
    ])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train per-ROI anomaly models on pre-cropped grayscale folders."
    )
    parser.add_argument(
        "--rois-root", type=Path, default=DEFAULT_ROIS_ROOT,
        help=f"Root directory containing ROI subfolders (default: {DEFAULT_ROIS_ROOT})"
    )
    parser.add_argument(
        "--models-root", type=Path, default=DEFAULT_MODELS_ROOT,
        help=f"Directory to save trained models (default: {DEFAULT_MODELS_ROOT})"
    )
    parser.add_argument(
        "--model", choices=["efficientad", "patchcore"], default="efficientad",
        help="Anomaly model type (default: efficientad)"
    )
    parser.add_argument(
        "--image-size", nargs=2, type=int, default=None, metavar=("H", "W"),
        help="Override image size, e.g. --image-size 256 256"
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Override max training epochs"
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="Override train batch size"
    )
    parser.add_argument(
        "--only", nargs="+", default=None, metavar="ROI_NAME",
        help="Train only these ROI folder names, e.g. --only ROI_LOGO ROI_BARCODE"
    )
    args = parser.parse_args()

    # ── Resolve training config ───────────────────────────────────────────────
    cfg = dict(TRAIN_CONFIG)
    if args.image_size:
        cfg["image_size"] = tuple(args.image_size)
    if args.epochs:
        cfg["max_epochs"] = args.epochs
    if args.batch_size:
        cfg["train_batch_size"] = args.batch_size

    rois_root   = args.rois_root
    models_root = args.models_root

    # ── Discover ROI folders ──────────────────────────────────────────────────
    if not rois_root.exists():
        print(f"\n[ERROR] ROI data root not found: {rois_root}")
        print("  Make sure the pre-cropped ROI folders are transferred to this path.")
        sys.exit(1)

    roi_folders = discover_roi_folders(rois_root)
    if not roi_folders:
        print(f"\n[ERROR] No ROI subfolders found in {rois_root}")
        print("  Each subfolder must have a 'good/' directory inside it.")
        sys.exit(1)

    if args.only:
        roi_folders = [f for f in roi_folders if f.name in args.only]
        if not roi_folders:
            print(f"[ERROR] None of {args.only} found in {rois_root}")
            sys.exit(1)

    # ── Pre-flight check ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  HIMALAYA NSC — PER-ROI ANOMALY MODEL TRAINING")
    print("=" * 60)
    print(f"\n  Data root:    {rois_root}")
    print(f"  Models root:  {models_root}")
    print(f"  Model type:   {args.model.upper()}")
    print(f"  Image size:   {cfg['image_size']}")
    print(f"  Epochs:       {cfg['max_epochs']}\n")
    print(f"  ROI folders found ({len(roi_folders)}):")

    all_ok = True
    for roi_dir in roi_folders:
        check = verify_roi_folder(roi_dir)
        status = "✓" if check["ok"] else "✗"
        print(f"    {status}  {roi_dir.name:30s}  "
              f"good={check['n_good']:3d}  bad={check['n_bad']:3d}")
        if not check["ok"]:
            all_ok = False
            for err in check["errors"]:
                print(f"         [!] {err}")

    if not all_ok:
        print("\n[ERROR] Some ROI folders are invalid. Fix above errors before training.")
        sys.exit(1)

    print(f"\n  Starting training for {len(roi_folders)} ROI(s)...\n")

    # ── Train each ROI ────────────────────────────────────────────────────────
    roi_model_paths = {}
    results = []
    total_start = time.time()

    for roi_dir in roi_folders:
        roi_name   = roi_dir.name
        output_dir = models_root / roi_name

        print(f"\n{'─' * 60}")
        print(f"  Training: {roi_name}")
        print(f"{'─' * 60}")

        t0 = time.time()
        try:
            saved_dir = train_roi_model(
                roi_data_dir=roi_dir,
                output_dir=output_dir,
                model_type=args.model,
                image_size=cfg["image_size"],
                max_epochs=cfg["max_epochs"],
                train_batch_size=cfg["train_batch_size"],
                num_workers=cfg["num_workers"],
                model_size=cfg["model_size"],
                coreset_sampling_ratio=cfg["coreset_sampling_ratio"],
                num_neighbors=cfg["num_neighbors"],
            )
            elapsed = time.time() - t0
            roi_model_paths[roi_name] = str(saved_dir)
            results.append({"roi": roi_name, "status": "OK", "elapsed_s": round(elapsed, 1)})
            print(f"\n  ✓ {roi_name} done in {elapsed:.0f}s")

        except Exception as exc:
            elapsed = time.time() - t0
            results.append({"roi": roi_name, "status": f"FAILED: {exc}", "elapsed_s": round(elapsed, 1)})
            print(f"\n  ✗ {roi_name} FAILED: {exc}")

    # ── Save roi_model_paths.json ─────────────────────────────────────────────
    models_root.mkdir(parents=True, exist_ok=True)
    paths_file = models_root / "roi_model_paths.json"
    with open(paths_file, "w") as f:
        json.dump(roi_model_paths, f, indent=2)

    # ── Final summary ─────────────────────────────────────────────────────────
    total_elapsed = time.time() - total_start
    n_ok   = sum(1 for r in results if r["status"] == "OK")
    n_fail = len(results) - n_ok

    print("\n\n" + "=" * 60)
    print("  TRAINING SUMMARY")
    print("=" * 60)
    for r in results:
        icon = "✓" if r["status"] == "OK" else "✗"
        print(f"  {icon}  {r['roi']:30s}  {r['elapsed_s']:6.0f}s  {r['status']}")
    print(f"\n  Total: {n_ok} OK  /  {n_fail} FAILED  "
          f"(total time: {total_elapsed/60:.1f} min)")
    print(f"\n  Model paths saved → {paths_file}")
    print("\nNext step:")
    print("  python scripts/calibrate_roi_thresholds.py")
    print("  (After Person C has finished annotating the bad crops)")


if __name__ == "__main__":
    main()
