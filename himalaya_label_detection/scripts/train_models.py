"""
train_models.py
───────────────
Phase 2 script — trains the PatchCore anomaly model on preprocessed data.

Run this on the TRAINING MACHINE after copying data/processed/ from Phase 1.

What it does:
  1. Optionally build and save the Golden Master EfficientNet-B0 embedding.
  2. Prepare anomalib-compatible dataset layout from Phase 1 output.
  3. Train PatchCore on the augmented good images (~2050 images).
  4. Run a test pass using bad images to estimate anomaly score distribution.

Usage:
    python scripts/train_models.py
    python scripts/train_models.py --build-gm-embedding
    python scripts/train_models.py --image-size 384 384
"""

import sys
import argparse
import yaml
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.training.train_patchcore import prepare_anomalib_dataset, train_patchcore
from src.utils.image_utils import list_images

PROJECT_ROOT    = Path(__file__).resolve().parent.parent
PIPELINE_CONFIG = PROJECT_ROOT / "config" / "pipeline_config.yaml"
GM_PATH         = PROJECT_ROOT / "golden_master" / "golden_master.jpg"


def load_config() -> dict:
    with open(PIPELINE_CONFIG) as f:
        return yaml.safe_load(f)


def check_prerequisites(cfg: dict) -> None:
    errors = []
    aug_dir = PROJECT_ROOT / cfg["paths"]["augmented_good"]
    if not aug_dir.exists() or not list_images(aug_dir):
        errors.append(
            f"No augmented images found in {aug_dir}\n"
            "  → Run Phase 1 first:  python scripts/run_preprocessing.py"
        )
    bad_dir = PROJECT_ROOT / cfg["paths"]["aligned_bad"]
    if not bad_dir.exists() or not list_images(bad_dir):
        errors.append(
            f"No aligned bad images found in {bad_dir}\n"
            "  → Run Phase 1 first:  python scripts/run_preprocessing.py"
        )
    if errors:
        print("\n[PREREQUISITES NOT MET]")
        for e in errors:
            print(f"  ✗ {e}")
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Phase 2 PatchCore anomaly model.")
    parser.add_argument("--build-gm-embedding", action="store_true",
                        help="Also build and save the Golden Master embedding")
    parser.add_argument("--image-size", nargs=2, type=int, default=None,
                        metavar=("H", "W"),
                        help="Override image size (default: from pipeline_config.yaml)")
    args = parser.parse_args()

    cfg = load_config()
    check_prerequisites(cfg)

    h = args.image_size[0] if args.image_size else cfg["image"]["height"]
    w = args.image_size[1] if args.image_size else cfg["image"]["width"]
    image_size = (h, w)

    print("\n" + "=" * 55)
    print("  HIMALAYA LABEL — PHASE 2: TRAINING")
    print("=" * 55)

    # ── Step 1: Golden Master embedding ────────────────────────────────────────
    if args.build_gm_embedding:
        print("\n[1/3] Building Golden Master embedding...")
        from src.verification.gm_check import GoldenMasterChecker
        if not GM_PATH.exists():
            print(f"  [SKIP] Golden master not found at {GM_PATH}")
            print("         Run: python scripts/setup_golden_master.py first.")
        else:
            checker = GoldenMasterChecker()
            checker.build_embedding(
                golden_master_path=GM_PATH,
                save_path=PROJECT_ROOT / "models" / "golden_master_embedding.pt",
            )
    else:
        print("\n[1/3] Golden Master embedding — skipped "
              "(use --build-gm-embedding to enable)")

    # ── Step 2: Prepare anomalib dataset ───────────────────────────────────────
    print("\n[2/3] Preparing anomalib dataset layout...")
    dataset_root = prepare_anomalib_dataset(
        augmented_good_dir=PROJECT_ROOT / cfg["paths"]["augmented_good"],
        aligned_bad_dir=   PROJECT_ROOT / cfg["paths"]["aligned_bad"],
        anomalib_dir=      PROJECT_ROOT / "data" / "anomalib_ready",
        dataset_name="full_label",
    )

    # ── Step 3: Train PatchCore ─────────────────────────────────────────────────
    print(f"\n[3/3] Training PatchCore (image size: {image_size})...")
    train_patchcore(
        dataset_root=dataset_root,
        dataset_name="full_label",
        output_dir=PROJECT_ROOT / "models" / "full_label",
        image_size=image_size,
    )

    print("\n" + "=" * 55)
    print("  TRAINING COMPLETE")
    print("=" * 55)
    print("\nNext steps:")
    print("  Calibrate thresholds:  python scripts/calibrate_thresholds.py")
    print("  Run inference:         python scripts/run_inference.py --folder data/raw/bad/")
    print("  Acceptance test:       python scripts/validate.py")


if __name__ == "__main__":
    main()
