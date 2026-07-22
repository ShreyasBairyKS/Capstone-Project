"""
organise_roi_folders.py
────────────────────────────────────────────────────────────────────────────
Helps Person C split the 3 mixed ROI folders into good/ and bad/ subfolders.

The 3 pre-cropped ROI folders on the remote PC currently look like:
    ROI_LOGO/          ← mixed: good and bad images together
    ROI_INGREDIENT/    ← mixed
    ROI_ADDRESS/       ← mixed

After running this script (with manual sorting), they will look like:
    ROI_LOGO/
        good/   ← images you confirmed as defect-free
        bad/    ← images with visible defects
    ROI_INGREDIENT/
        good/
        bad/
    ROI_ADDRESS/
        good/
        bad/

HOW TO USE
──────────
Step 1: Run this script to create the subfolder structure:
    python organise_roi_folders.py --create-structure

Step 2: Open each ROI folder and MANUALLY sort images:
    - Move clearly GOOD images → good/
    - Move clearly BAD images  → bad/
    - Use your eyes — look for tears, smudges, missing print, ink blobs
    
Step 3: Run again to verify counts:
    python organise_roi_folders.py --verify

Usage:
    python organise_roi_folders.py --rois-root /path/to/roi/folders --create-structure
    python organise_roi_folders.py --rois-root /path/to/roi/folders --verify
"""

import argparse
import shutil
from pathlib import Path

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}


def create_structure(rois_root: Path):
    """Create good/ and bad/ subfolders inside each ROI folder."""
    roi_dirs = [d for d in rois_root.iterdir() if d.is_dir()
                and d.name not in ("good", "bad")]

    if not roi_dirs:
        print(f"No ROI folders found in {rois_root}")
        return

    print(f"Found {len(roi_dirs)} ROI folders:")
    for roi_dir in sorted(roi_dirs):
        good_dir = roi_dir / "good"
        bad_dir  = roi_dir / "bad"
        good_dir.mkdir(exist_ok=True)
        bad_dir.mkdir(exist_ok=True)

        # Count images already directly in this folder (not in subfolders yet)
        mixed_images = [f for f in roi_dir.iterdir()
                        if f.is_file() and f.suffix.lower() in IMAGE_EXTS]

        print(f"\n  {roi_dir.name}/")
        print(f"    good/  ← created")
        print(f"    bad/   ← created")
        if mixed_images:
            print(f"    {len(mixed_images)} images still in root (not sorted yet)")
            print(f"    → Manually move them into good/ or bad/")
        else:
            print(f"    No unsorted images found in root")


def verify(rois_root: Path):
    """Print counts and flag any unsorted images."""
    roi_dirs = [d for d in rois_root.iterdir() if d.is_dir()
                and d.name not in ("good", "bad")]

    print(f"\n{'='*55}")
    print(f"  ROI FOLDER VERIFICATION")
    print(f"{'='*55}")

    all_ok = True
    for roi_dir in sorted(roi_dirs):
        good_dir = roi_dir / "good"
        bad_dir  = roi_dir / "bad"

        good_count = len([f for f in good_dir.glob("*")
                          if f.suffix.lower() in IMAGE_EXTS]) if good_dir.exists() else 0
        bad_count  = len([f for f in bad_dir.glob("*")
                          if f.suffix.lower() in IMAGE_EXTS]) if bad_dir.exists() else 0
        unsorted   = [f for f in roi_dir.iterdir()
                      if f.is_file() and f.suffix.lower() in IMAGE_EXTS]

        status = "✓" if (good_count > 0 and bad_count > 0 and not unsorted) else "⚠"
        print(f"\n  {status}  {roi_dir.name}/")
        print(f"       good/ : {good_count} images")
        print(f"       bad/  : {bad_count} images")
        if unsorted:
            print(f"       ⚠ {len(unsorted)} UNSORTED images still in root — please move them!")
            all_ok = False

    print()
    if all_ok:
        print("  ✅ All folders correctly organised. Ready for training.")
    else:
        print("  ⚠️  Some folders have unsorted images. Sort them before training.")
    print()


def main():
    parser = argparse.ArgumentParser(description="Organise ROI folders into good/ and bad/")
    parser.add_argument("--rois-root",        type=Path, default=Path("data/rois"),
                        help="Path to folder containing ROI_LOGO, ROI_INGREDIENT, ROI_ADDRESS")
    parser.add_argument("--create-structure", action="store_true",
                        help="Create good/ and bad/ subfolders")
    parser.add_argument("--verify",           action="store_true",
                        help="Print counts and check for unsorted images")
    args = parser.parse_args()

    rois_root = args.rois_root
    if not rois_root.exists():
        print(f"ERROR: {rois_root} does not exist")
        print(f"Set --rois-root to the folder containing ROI_LOGO, ROI_INGREDIENT, ROI_ADDRESS")
        return

    if args.create_structure:
        create_structure(rois_root)
        print("\nNext: open each ROI folder in File Explorer and manually sort images")
        print("Then run:  python organise_roi_folders.py --verify")

    if args.verify:
        verify(rois_root)

    if not args.create_structure and not args.verify:
        parser.print_help()


if __name__ == "__main__":
    main()
