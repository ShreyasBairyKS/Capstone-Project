"""
run_preprocessing.py
────────────────────
Main Phase 1 pipeline script. Runs everything end-to-end:

  1. Align all good and bad images to the golden master (ORB + Homography).
  2. Save aligned images to data/processed/aligned/{good,bad}/
  3. Extract functional ROI crops from every aligned image.
  4. Save ROI crops to data/processed/rois/{ROI_NAME}/{good,bad}/
  5. Generate augmented versions of the aligned GOOD images only.
  6. Save augmented images to data/processed/augmented/good/
  7. Write a summary report to data/processed/preprocessing_report.txt

What gets copied to the training machine:
    data/processed/
    ├── aligned/good/        ← 80 aligned good images  (for PatchCore memory bank)
    ├── aligned/bad/         ← 50 aligned bad images   (for threshold calibration)
    ├── augmented/good/      ← ~2000 augmented images  (for training)
    └── rois/                ← ROI crops                (for future per-ROI models)

Usage:
    cd himalaya_label_detection
    python scripts/run_preprocessing.py

    # Skip augmentation (faster, for testing):
    python scripts/run_preprocessing.py --skip-augmentation

    # Skip ROI extraction:
    python scripts/run_preprocessing.py --skip-rois
"""

import sys
import argparse
import time
import yaml
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessing.align import LabelAligner, AlignmentResult
from src.preprocessing.roi_extractor import ROIExtractor
from src.preprocessing.augment import ImageAugmentor
from src.utils.image_utils import (
    load_image, save_image, list_images, to_grayscale
)
from tqdm import tqdm

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT    = Path(__file__).resolve().parent.parent
ROI_CONFIG_PATH = PROJECT_ROOT / "config" / "roi_config.yaml"
PIPELINE_CONFIG = PROJECT_ROOT / "config" / "pipeline_config.yaml"
GM_PATH         = PROJECT_ROOT / "golden_master" / "golden_master.jpg"


def load_config() -> dict:
    with open(PIPELINE_CONFIG) as f:
        return yaml.safe_load(f)


def check_prerequisites(cfg: dict) -> None:
    """Abort with a clear message if setup has not been done."""
    errors = []

    if not GM_PATH.exists():
        errors.append(
            f"Golden master not found at {GM_PATH}\n"
            "  → Run: python scripts/setup_golden_master.py"
        )

    with open(ROI_CONFIG_PATH) as f:
        roi_cfg = yaml.safe_load(f)
    if not roi_cfg.get("configured", False):
        errors.append(
            "ROI config not set up.\n"
            "  → Run: python scripts/setup_golden_master.py"
        )

    good_dir = PROJECT_ROOT / cfg["paths"]["raw_good"]
    if not good_dir.exists() or not list_images(good_dir):
        errors.append(f"No good images found in {good_dir}")

    if errors:
        print("\n[SETUP ERRORS]")
        for e in errors:
            print(f"  ✗ {e}")
        sys.exit(1)


def align_split(split: str,
                aligner: LabelAligner,
                cfg: dict,
                target_size: tuple) -> tuple[list[Path], list[str]]:
    """
    Align all images in one split (good or bad).

    Returns:
        (aligned_paths, failed_names)  — paths of saved aligned images
    """
    src_dir  = PROJECT_ROOT / cfg["paths"][f"raw_{split}"]
    dest_dir = PROJECT_ROOT / cfg["paths"][f"aligned_{split}"]
    dest_dir.mkdir(parents=True, exist_ok=True)

    images = list_images(src_dir)
    if not images:
        print(f"  [SKIP] No images in {src_dir}")
        return [], []

    aligned_paths = []
    failed = []

    for img_path in tqdm(images, desc=f"Aligning {split}"):
        try:
            raw = load_image(img_path, target_size=target_size)
            result: AlignmentResult = aligner.align(raw)

            if not result.success:
                failed.append(img_path.name)
                # Still save the best-effort aligned image — don't silently discard
                tqdm.write(f"  [WARN] Low inliers ({result.num_inliers}) for {img_path.name}")

            out_path = dest_dir / img_path.name
            save_image(result.image, out_path)
            aligned_paths.append(out_path)

        except Exception as exc:
            failed.append(img_path.name)
            tqdm.write(f"  [ERROR] {img_path.name}: {exc}")

    return aligned_paths, failed


def extract_rois(aligned_paths: list[Path],
                 split: str,
                 extractor: ROIExtractor,
                 rois_root: Path) -> dict:
    """
    Extract ROI crops from aligned images and save per-ROI per-split.

    Directory layout:
        rois_root/ROI_LOGO/good/img_001.jpg
        rois_root/ROI_LOGO/bad/img_001.jpg
        ...

    Returns:
        stats dict with counts per ROI.
    """
    stats = {}

    for img_path in tqdm(aligned_paths, desc=f"Extracting ROIs ({split})"):
        image = load_image(img_path)
        result = extractor.extract(image)

        for roi_name, crop in result.crops.items():
            if not result.valid.get(roi_name, False):
                continue
            out_dir = rois_root / roi_name / split
            out_dir.mkdir(parents=True, exist_ok=True)
            save_image(crop, out_dir / img_path.name)
            stats[roi_name] = stats.get(roi_name, 0) + 1

    return stats


def write_report(report_path: Path,
                 n_good_raw: int, n_bad_raw: int,
                 n_good_aligned: int, n_bad_aligned: int,
                 n_good_failed: int, n_bad_failed: int,
                 good_failed_names: list[str], bad_failed_names: list[str],
                 n_augmented: int,
                 roi_stats: dict,
                 elapsed: float) -> None:

    lines = [
        "=" * 60,
        "HIMALAYA LABEL PREPROCESSING REPORT",
        "=" * 60,
        "",
        "── Input ──────────────────────────────────────────────────",
        f"  Good images (raw):       {n_good_raw}",
        f"  Bad  images (raw):       {n_bad_raw}",
        "",
        "── Alignment ──────────────────────────────────────────────",
        f"  Good aligned (saved):   {n_good_aligned}",
        f"  Bad  aligned (saved):   {n_bad_aligned}",
        f"  Good alignment failures: {n_good_failed}",
        f"  Bad  alignment failures: {n_bad_failed}",
    ]
    if good_failed_names:
        lines.append("  Failed good images:")
        lines += [f"    - {n}" for n in good_failed_names]
    if bad_failed_names:
        lines.append("  Failed bad images:")
        lines += [f"    - {n}" for n in bad_failed_names]

    lines += [
        "",
        "── Augmentation ───────────────────────────────────────────",
        f"  Augmented good images:  {n_augmented}",
        "",
        "── ROI Extraction ─────────────────────────────────────────",
    ]
    for roi, count in sorted(roi_stats.items()):
        lines.append(f"  {roi:<28} {count:>4} crops saved")

    lines += [
        "",
        "── Performance ────────────────────────────────────────────",
        f"  Total time:  {elapsed:.1f}s",
        "",
        "── Output layout ───────────────────────────────────────────",
        "  data/processed/",
        "  ├── aligned/good/        ← for PatchCore memory bank",
        "  ├── aligned/bad/         ← for threshold calibration",
        "  ├── augmented/good/      ← for model training",
        "  └── rois/<ROI_NAME>/     ← for future per-ROI models",
        "",
        "=" * 60,
    ]

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines))
    print("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 1 preprocessing pipeline.")
    parser.add_argument("--skip-augmentation", action="store_true",
                        help="Skip augmentation step (useful for testing)")
    parser.add_argument("--skip-rois", action="store_true",
                        help="Skip ROI extraction step")
    args = parser.parse_args()

    cfg = load_config()
    check_prerequisites(cfg)

    target_size = (cfg["image"]["width"], cfg["image"]["height"])

    print("\n" + "=" * 50)
    print("  HIMALAYA LABEL PREPROCESSING — PHASE 1")
    print("=" * 50 + "\n")

    start = time.time()

    # ── Step 1: Build aligner ────────────────────────────────────────────────
    print("[1/4] Initialising aligner...")
    aligner = LabelAligner(
        golden_master_path=GM_PATH,
        target_size=target_size,
        orb_max_features=cfg["alignment"]["orb_max_features"],
        match_keep_top=cfg["alignment"]["match_keep_top"],
        ransac_threshold=cfg["alignment"]["ransac_threshold"],
        min_match_count=cfg["alignment"]["min_match_count"],
    )
    print(f"      Golden master loaded: {GM_PATH.name}")

    # ── Step 2: Align good and bad images ─────────────────────────────────────
    print("\n[2/4] Aligning images...")
    good_raw_paths = list_images(PROJECT_ROOT / cfg["paths"]["raw_good"])
    bad_raw_paths  = list_images(PROJECT_ROOT / cfg["paths"]["raw_bad"])

    aligned_good, good_failed = align_split("good", aligner, cfg, target_size)
    aligned_bad,  bad_failed  = align_split("bad",  aligner, cfg, target_size)

    # ── Step 3: Extract ROIs ─────────────────────────────────────────────────
    roi_stats = {}
    if not args.skip_rois:
        print("\n[3/4] Extracting functional ROIs...")
        extractor = ROIExtractor(ROI_CONFIG_PATH)
        rois_root = PROJECT_ROOT / cfg["paths"]["rois_root"]
        stats_good = extract_rois(aligned_good, "good", extractor, rois_root)
        stats_bad  = extract_rois(aligned_bad,  "bad",  extractor, rois_root)
        # Merge counts
        for k, v in {**stats_good, **stats_bad}.items():
            roi_stats[k] = roi_stats.get(k, 0) + v
    else:
        print("\n[3/4] ROI extraction skipped (--skip-rois)")

    # ── Step 4: Augment good images ───────────────────────────────────────────
    n_augmented = 0
    if not args.skip_augmentation:
        print("\n[4/4] Augmenting good images...")
        augmentor = ImageAugmentor(
            num_per_image=cfg["augmentation"]["num_augmentations_per_image"]
        )
        aug_good_dir = PROJECT_ROOT / cfg["paths"]["augmented_good"]
        aligned_good_dir = PROJECT_ROOT / cfg["paths"]["aligned_good"]
        n_augmented = augmentor.augment_directory(
            input_dir=aligned_good_dir,
            output_dir=aug_good_dir,
            copy_originals=cfg["augmentation"]["save_original_in_augmented"]
        )
        print(f"      {n_augmented} images saved to {aug_good_dir}")
    else:
        print("\n[4/4] Augmentation skipped (--skip-augmentation)")

    # ── Report ─────────────────────────────────────────────────────────────────
    elapsed = time.time() - start
    report_path = PROJECT_ROOT / "data" / "processed" / "preprocessing_report.txt"
    write_report(
        report_path,
        n_good_raw=len(good_raw_paths),
        n_bad_raw=len(bad_raw_paths),
        n_good_aligned=len(aligned_good),
        n_bad_aligned=len(aligned_bad),
        n_good_failed=len(good_failed),
        n_bad_failed=len(bad_failed),
        good_failed_names=good_failed,
        bad_failed_names=bad_failed,
        n_augmented=n_augmented,
        roi_stats=roi_stats,
        elapsed=elapsed,
    )
    print(f"\nReport saved to: {report_path}")
    print("\n[DONE] Phase 1 complete. Copy data/processed/ to your training machine.\n")


if __name__ == "__main__":
    main()
