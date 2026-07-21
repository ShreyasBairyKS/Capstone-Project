"""
verify_alignment.py
───────────────────
Visual sanity check for the ORB + Homography alignment step.

Shows side-by-side comparisons and a difference heatmap so you can
confirm the alignment quality before running the full preprocessing.

Usage:
    cd himalaya_label_detection
    python scripts/verify_alignment.py                 # checks 5 random good images
    python scripts/verify_alignment.py --n 10          # checks 10 images
    python scripts/verify_alignment.py --all           # checks all images
    python scripts/verify_alignment.py --bad           # also checks bad images

Controls:
    SPACE / ENTER  → next image
    'f'            → flag current image (printed to console at end)
    ESC / 'q'      → quit early
"""

import sys
import argparse
import random
import cv2
import numpy as np
import yaml
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessing.align import LabelAligner, AlignmentError
from src.utils.image_utils import load_image, list_images, side_by_side, diff_map

PROJECT_ROOT    = Path(__file__).resolve().parent.parent
GM_PATH         = PROJECT_ROOT / "golden_master" / "golden_master.jpg"
PIPELINE_CONFIG = PROJECT_ROOT / "config" / "pipeline_config.yaml"
GOOD_DIR        = PROJECT_ROOT / "data" / "raw" / "good"
BAD_DIR         = PROJECT_ROOT / "data" / "raw" / "bad"


def load_pipeline_config() -> dict:
    with open(PIPELINE_CONFIG) as f:
        return yaml.safe_load(f)


def check_prerequisites() -> None:
    if not GM_PATH.exists():
        print("[ERROR] Golden master not found.")
        print("        Run: python scripts/setup_golden_master.py first.")
        sys.exit(1)
    if not GOOD_DIR.exists() or not list_images(GOOD_DIR):
        print(f"[ERROR] No images found in {GOOD_DIR}")
        sys.exit(1)


def show_alignment(aligner: LabelAligner,
                   image_path: Path,
                   target_size: tuple) -> str:
    """
    Align one image and display the comparison window.

    Returns:
        'next'   → user pressed SPACE/ENTER
        'flag'   → user pressed 'f'
        'quit'   → user pressed ESC or 'q'
    """
    image = load_image(image_path, target_size=target_size)
    result = aligner.align(image)

    # Build display panels
    golden = load_image(GM_PATH, target_size=target_size)
    panel_compare = side_by_side(
        golden, result.image,
        label_a="Golden Master",
        label_b=f"Aligned ({result.num_inliers} inliers)"
    )
    panel_diff = diff_map(golden, result.image)

    # Stack compare on top, diff below
    diff_resized = cv2.resize(panel_diff, (panel_compare.shape[1], panel_compare.shape[0] // 3))
    status_color = (0, 200, 0) if result.success else (0, 0, 220)
    status_text  = "OK" if result.success else "FAIL — low inliers"

    # Header bar
    header = np.zeros((36, panel_compare.shape[1], 3), dtype=np.uint8)
    cv2.putText(header,
                f"{image_path.name}  |  {status_text}  |  inliers: {result.num_inliers}  "
                f"|  [SPACE=next] [f=flag] [ESC=quit]",
                (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.52, status_color, 1, cv2.LINE_AA)

    display = np.vstack([header, panel_compare, diff_resized])

    win = "Alignment Verification"
    cv2.imshow(win, display)

    while True:
        key = cv2.waitKey(0) & 0xFF
        if key in (32, 13):      # SPACE or ENTER
            return "next"
        if key == ord("f"):
            return "flag"
        if key in (27, ord("q")):  # ESC or 'q'
            return "quit"


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify label alignment quality.")
    parser.add_argument("--n",   type=int,  default=5,     help="Number of images to check")
    parser.add_argument("--all", action="store_true",      help="Check all images")
    parser.add_argument("--bad", action="store_true",      help="Also include bad images")
    args = parser.parse_args()

    check_prerequisites()
    cfg = load_pipeline_config()
    target_size = (cfg["image"]["width"], cfg["image"]["height"])

    aligner = LabelAligner(
        golden_master_path=GM_PATH,
        target_size=target_size,
        orb_max_features=cfg["alignment"]["orb_max_features"],
        match_keep_top=cfg["alignment"]["match_keep_top"],
        ransac_threshold=cfg["alignment"]["ransac_threshold"],
        min_match_count=cfg["alignment"]["min_match_count"],
    )

    # Collect image paths
    all_paths = list_images(GOOD_DIR)
    if args.bad:
        all_paths += list_images(BAD_DIR)

    if not args.all:
        n = min(args.n, len(all_paths))
        all_paths = random.sample(all_paths, n)
        print(f"Checking {n} randomly sampled images.")
    else:
        print(f"Checking all {len(all_paths)} images.")

    flagged = []

    for i, img_path in enumerate(all_paths):
        print(f"[{i+1}/{len(all_paths)}] {img_path.name}", end="  ")
        action = show_alignment(aligner, img_path, target_size)
        if action == "flag":
            flagged.append(img_path.name)
            print("→ FLAGGED")
        elif action == "quit":
            print("→ Quit by user.")
            break
        else:
            print("→ OK")

    cv2.destroyAllWindows()

    print("\n=== Verification Complete ===")
    if flagged:
        print(f"Flagged images ({len(flagged)}):")
        for name in flagged:
            print(f"  - {name}")
        print("Review these images — poor alignment may indicate:")
        print("  · Insufficient texture for ORB (adjust lighting)")
        print("  · Image too blurry (check focus / shutter speed)")
        print("  · Try increasing orb_max_features in pipeline_config.yaml")
    else:
        print("No images flagged. Alignment looks good.")
        print("You can now run:  python scripts/run_preprocessing.py")


if __name__ == "__main__":
    main()
