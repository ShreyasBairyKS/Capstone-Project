"""
crop_roi.py  —  Interactive ROI Cropping Tool
═══════════════════════════════════════════════════════════════════════════════
Himalaya NSC Label Inspection Project
Used by all 3 team members to crop their assigned ROI from 132 full images.

HOW IT WORKS
─────────────
• Opens each full image in a scrollable window (image is 8000px tall)
• You drag a rectangle around the complete ROI region
• Press G = save to good/   B = save to bad/   S = skip   Q = quit

CYLINDRICAL SPLIT RULE (IMPORTANT)
────────────────────────────────────
The label wraps around the tube, so each region appears TWICE per image.
Sometimes one occurrence is cut/incomplete at the image edge.
→ ALWAYS select the COMPLETE, uncut occurrence.
→ If both are cut, press S to skip that image.
→ You can scroll to see the full image height.

USAGE
──────
python crop_roi.py --roi ROI_1 --out-dir data/rois/ROI_1

CONTROLS IN THE WINDOW
────────────────────────
  Click + drag   = draw crop rectangle (RED box)
  G              = save crop to good/ subfolder
  B              = save crop to bad/  subfolder
  S              = skip this image (both occurrences cut/incomplete)
  R              = reset / redraw rectangle
  Q              = quit and save progress

ARGUMENTS
──────────
  --roi        ROI name for display (e.g. ROI_1)
  --images     path to folder with full BMP images [default: dataset/NSC]
  --out-dir    where to save good/ and bad/ subfolders
  --start-from filename to resume from (e.g. 45.bmp)
"""

import argparse
import json
import cv2
import numpy as np
from pathlib import Path

# ── Globals for mouse callback ─────────────────────────────────────────────────
drawing = False
ix, iy, ex, ey = 0, 0, 0, 0
rect_done = False
display_img = None   # the image currently shown in window
scale = 1.0          # display scale factor


def mouse_callback(event, x, y, flags, param):
    global drawing, ix, iy, ex, ey, rect_done, display_img

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        rect_done = False
        ix, iy = x, y
        ex, ey = x, y

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        ex, ey = x, y
        temp = display_img.copy()
        cv2.rectangle(temp, (ix, iy), (ex, ey), (0, 0, 255), 2)
        cv2.imshow("NSC Crop Tool", temp)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        rect_done = True
        ex, ey = x, y
        temp = display_img.copy()
        cv2.rectangle(temp, (ix, iy), (ex, ey), (0, 255, 0), 2)
        status = "[G]=save good  [B]=save bad  [S]=skip  [R]=redo"
        cv2.putText(temp, status, (10, temp.shape[0]-15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
        cv2.imshow("NSC Crop Tool", temp)


def get_display_image(full_img: np.ndarray, max_h: int = 900) -> tuple[np.ndarray, float]:
    """Resize image to fit screen height, return (display_img, scale)."""
    h, w = full_img.shape[:2]
    sc = min(1.0, max_h / h)
    display = cv2.resize(full_img, (int(w * sc), int(h * sc)))
    return display, sc


def crop_and_save(full_img: np.ndarray, x1d, y1d, x2d, y2d,
                  scale: float, out_path: Path):
    """Convert display coords back to full-image coords and save crop."""
    x1 = int(min(x1d, x2d) / scale)
    y1 = int(min(y1d, y2d) / scale)
    x2 = int(max(x1d, x2d) / scale)
    y2 = int(max(y1d, y2d) / scale)

    # Clamp to image bounds
    h, w = full_img.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    crop = full_img[y1:y2, x1:x2]
    if crop.size == 0:
        return False, (0, 0, 0, 0)

    cv2.imwrite(str(out_path), crop)
    return True, (x1, y1, x2 - x1, y2 - y1)


def load_progress(progress_file: Path) -> set:
    """Load set of already-processed filenames."""
    if progress_file.exists():
        with open(progress_file) as f:
            data = json.load(f)
        return set(data.get("done", []))
    return set()


def save_progress(progress_file: Path, done: set):
    with open(progress_file, "w") as f:
        json.dump({"done": sorted(done)}, f, indent=2)


def main():
    global display_img, scale, rect_done, ix, iy, ex, ey

    parser = argparse.ArgumentParser(description="Interactive ROI crop tool")
    parser.add_argument("--roi",        default="ROI_1",
                        help="ROI name, e.g. ROI_1")
    parser.add_argument("--images",     default="dataset/NSC",
                        help="Folder containing NSC GOOD IMAGES/ and NSC BAD IMAGES/")
    parser.add_argument("--out-dir",    default=None,
                        help="Output folder. Default: data/rois/<roi>/")
    parser.add_argument("--start-from", default=None,
                        help="Resume from this filename, e.g. 45.bmp")
    parser.add_argument("--good-only",  action="store_true",
                        help="Process only good images")
    parser.add_argument("--bad-only",   action="store_true",
                        help="Process only bad images")
    args = parser.parse_args()

    roi_name = args.roi
    images_root = Path(args.images)
    out_dir = Path(args.out_dir) if args.out_dir else Path(f"data/rois/{roi_name}")
    good_dir = out_dir / "good"
    bad_dir  = out_dir / "bad"
    good_dir.mkdir(parents=True, exist_ok=True)
    bad_dir.mkdir(parents=True, exist_ok=True)

    progress_file = out_dir / "progress.json"
    done = load_progress(progress_file)

    # Collect images
    folders = []
    if not args.bad_only:
        folders.append((images_root / "NSC GOOD IMAGES", "GOOD"))
    if not args.good_only:
        folders.append((images_root / "NSC BAD IMAGES", "BAD"))

    all_files = []
    for folder, label in folders:
        if folder.exists():
            for f in sorted(folder.glob("*.bmp")):
                all_files.append((f, label))
        else:
            print(f"[WARN] Not found: {folder}")

    if not all_files:
        print(f"No images found in {images_root}")
        print("Set --images to the folder containing 'NSC GOOD IMAGES' and 'NSC BAD IMAGES'")
        return

    # Resume support
    if args.start_from:
        start_names = [f.name for f, _ in all_files]
        if args.start_from in start_names:
            idx = start_names.index(args.start_from)
            all_files = all_files[idx:]
            print(f"Resuming from {args.start_from}")

    remaining = [(f, lbl) for f, lbl in all_files if f.name not in done]
    total = len(all_files)
    print(f"\n{'='*60}")
    print(f"  {roi_name} Cropping Tool")
    print(f"  {len(remaining)} images remaining  ({len(done)} already done / {total} total)")
    print(f"  Output: {out_dir}/")
    print(f"{'='*60}")
    print()
    print("  Controls:")
    print("    Click+drag  = draw selection box")
    print("    G           = save as GOOD crop")
    print("    B           = save as BAD  crop")
    print("    S           = skip (both occurrences incomplete)")
    print("    R           = reset selection")
    print("    Q           = quit (progress saved)")
    print()

    # Create window
    cv2.namedWindow("NSC Crop Tool", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("NSC Crop Tool", 800, 900)
    cv2.setMouseCallback("NSC Crop Tool", mouse_callback)

    good_count = len(list(good_dir.glob("*")))
    bad_count  = len(list(bad_dir.glob("*")))

    for file_idx, (img_path, orig_label) in enumerate(remaining):
        full_img = cv2.imread(str(img_path))
        if full_img is None:
            print(f"  ERROR: cannot read {img_path.name}")
            done.add(img_path.name)
            continue

        display_img, scale = get_display_image(full_img, max_h=900)
        rect_done = False
        ix = iy = ex = ey = 0

        # Draw header on display image
        header = display_img.copy()
        progress_pct = int(100 * file_idx / max(1, len(remaining)))
        info = (f"{roi_name}  |  {img_path.name}  [{orig_label}]  "
                f"({file_idx+1}/{len(remaining)})  {progress_pct}%  "
                f"good={good_count} bad={bad_count}")
        cv2.putText(header, info, (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
        inst = "Drag to select ROI. SCROLL if needed. G=good B=bad S=skip R=redo Q=quit"
        cv2.putText(header, inst, (10, 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        display_img = header
        cv2.imshow("NSC Crop Tool", display_img)

        action = None
        while True:
            key = cv2.waitKey(20) & 0xFF

            if key == ord('g') and rect_done:
                action = 'good'
                break
            elif key == ord('b') and rect_done:
                action = 'bad'
                break
            elif key == ord('s'):
                action = 'skip'
                break
            elif key == ord('r'):
                # Reset
                rect_done = False
                ix = iy = ex = ey = 0
                display_img, scale = get_display_image(full_img, max_h=900)
                cv2.imshow("NSC Crop Tool", display_img)
            elif key == ord('q'):
                action = 'quit'
                break

        if action == 'quit':
            print(f"\n  Quitting. Progress saved ({len(done)} done).")
            save_progress(progress_file, done)
            break

        if action == 'skip':
            print(f"  SKIP  {img_path.name}")
            done.add(img_path.name)
            save_progress(progress_file, done)
            continue

        # Save crop
        stem = img_path.stem
        out_subdir = good_dir if action == 'good' else bad_dir
        out_path = out_subdir / f"{stem}.png"
        success, (x, y, w, h) = crop_and_save(full_img, ix, iy, ex, ey, scale, out_path)

        if success:
            if action == 'good':
                good_count += 1
            else:
                bad_count += 1
            print(f"  {action.upper():4s}  {img_path.name} → {out_path.name}  (crop {w}x{h} at x={x},y={y})")
        else:
            print(f"  ERROR: empty crop for {img_path.name} — try again with --start-from {img_path.name}")

        done.add(img_path.name)
        save_progress(progress_file, done)

    cv2.destroyAllWindows()
    print(f"\nDone! good={good_count}  bad={bad_count}")
    print(f"Crops saved in: {out_dir}/")


if __name__ == "__main__":
    main()
