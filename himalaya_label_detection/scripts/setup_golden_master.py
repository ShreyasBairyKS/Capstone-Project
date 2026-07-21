"""
setup_golden_master.py
─────────────────────
Interactive script — run this ONCE before any other script.

What it does:
  1. Lets you pick which image from data/raw/good/ to use as the golden master.
  2. Copies it to golden_master/golden_master.jpg.
  3. Opens an interactive window where you draw a bounding box for each
     functional ROI using OpenCV's built-in selectROI tool.
  4. Saves the coordinates to config/roi_config.yaml.

Usage:
    cd himalaya_label_detection
    python scripts/setup_golden_master.py

Controls in selectROI window:
    Draw box  → click and drag
    Confirm   → press SPACE or ENTER
    Skip ROI  → press 'c' (use the placeholder coords for now)
    Quit      → press ESC (aborts without saving)
"""

import sys
import shutil
import cv2
import yaml
import numpy as np
from pathlib import Path

# Add project root to path so src imports work
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.image_utils import load_image, list_images, save_image

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT    = Path(__file__).resolve().parent.parent
ROI_CONFIG_PATH = PROJECT_ROOT / "config" / "roi_config.yaml"
GM_DIR          = PROJECT_ROOT / "golden_master"
GM_PATH         = GM_DIR / "golden_master.jpg"
GOOD_DIR        = PROJECT_ROOT / "data" / "raw" / "good"
PIPELINE_CONFIG = PROJECT_ROOT / "config" / "pipeline_config.yaml"


def load_pipeline_config() -> dict:
    with open(PIPELINE_CONFIG) as f:
        return yaml.safe_load(f)


def pick_golden_master() -> Path:
    """Let the user choose which good image becomes the golden master."""
    images = list_images(GOOD_DIR)
    if not images:
        print(f"[ERROR] No images found in {GOOD_DIR}")
        print("        Place your good label images there first, then re-run.")
        sys.exit(1)

    print("\n=== Golden Master Selection ===")
    print("Good images found:")
    for i, p in enumerate(images):
        print(f"  [{i}] {p.name}")

    while True:
        raw = input(f"\nEnter index [0–{len(images)-1}] or press ENTER for 0: ").strip()
        if raw == "":
            idx = 0
            break
        if raw.isdigit() and 0 <= int(raw) < len(images):
            idx = int(raw)
            break
        print("  Invalid choice, try again.")

    chosen = images[idx]
    print(f"\nSelected: {chosen.name}")
    return chosen


def draw_rois_interactively(image: np.ndarray,
                             roi_config: dict,
                             pipeline_cfg: dict) -> dict:
    """
    For each ROI defined in roi_config, open a selectROI window so the
    user can draw the bounding box on the golden master image.

    Returns updated roi_config with new coordinates.
    """
    target_w = pipeline_cfg["image"]["width"]
    target_h = pipeline_cfg["image"]["height"]
    display = cv2.resize(image.copy(), (target_w, target_h), interpolation=cv2.INTER_AREA)

    roi_names = list(roi_config["rois"].keys())
    total = len(roi_names)

    print("\n=== ROI Bounding Box Setup ===")
    print(f"You will draw {total} ROI boxes on the golden master image.")
    print("Controls: drag to draw → SPACE/ENTER to confirm → 'c' to skip")
    print("Press any key to start...\n")

    # Show a preview first
    preview = display.copy()
    cv2.putText(preview, "Golden Master — press any key to start ROI setup",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.imshow("Golden Master Preview", preview)
    cv2.waitKey(0)
    cv2.destroyWindow("Golden Master Preview")

    for i, roi_name in enumerate(roi_names):
        desc = roi_config["rois"][roi_name].get("description", "")
        print(f"[{i+1}/{total}] Draw box for: {roi_name}")
        print(f"       {desc}")

        # Draw all already-configured ROIs on the canvas for reference
        canvas = display.copy()
        cv2.putText(canvas, f"Drawing: {roi_name}  ({i+1}/{total})",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2)

        # Show previously confirmed ROIs in green
        for j, prev_name in enumerate(roi_names[:i]):
            prev = roi_config["rois"][prev_name]
            if prev["coords"] != [0, 0, 100, 100]:
                x, y, w, h = prev["coords"]
                cv2.rectangle(canvas, (x, y), (x+w, y+h), (0, 200, 0), 1)
                cv2.putText(canvas, prev_name.replace("ROI_", ""),
                            (x+2, y-4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 200, 0), 1)

        roi = cv2.selectROI(
            f"Setup — {roi_name}",
            canvas,
            fromCenter=False,
            showCrosshair=True
        )
        cv2.destroyWindow(f"Setup — {roi_name}")

        x, y, w, h = int(roi[0]), int(roi[1]), int(roi[2]), int(roi[3])

        if w == 0 or h == 0:
            print(f"  → Skipped (keeping placeholder coords)")
        else:
            roi_config["rois"][roi_name]["coords"] = [x, y, w, h]
            print(f"  → Saved: x={x}, y={y}, w={w}, h={h}")

    cv2.destroyAllWindows()
    return roi_config


def verify_rois(image: np.ndarray, roi_config: dict) -> None:
    """Show all ROIs drawn on the golden master for final confirmation."""
    canvas = image.copy()
    for roi_name, attrs in roi_config["rois"].items():
        x, y, w, h = attrs["coords"]
        if [x, y, w, h] == [0, 0, 100, 100]:
            continue
        color = (0, 0, 220) if attrs.get("critical") else (0, 200, 0)
        cv2.rectangle(canvas, (x, y), (x+w, y+h), color, 2)
        cv2.putText(canvas, roi_name.replace("ROI_", ""),
                    (x+3, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    cv2.putText(canvas, "Final ROI layout — press any key to save",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)
    cv2.imshow("ROI Verification", canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main() -> None:
    pipeline_cfg = load_pipeline_config()

    # Step 1: Pick golden master
    gm_source = pick_golden_master()

    # Step 2: Copy to golden_master/
    GM_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(gm_source, GM_PATH)
    print(f"\nGolden master saved to: {GM_PATH}")

    # Step 3: Load and resize for annotation
    target_w = pipeline_cfg["image"]["width"]
    target_h = pipeline_cfg["image"]["height"]
    gm_image = load_image(GM_PATH, target_size=(target_w, target_h))

    # Step 4: Load current ROI config
    with open(ROI_CONFIG_PATH) as f:
        roi_config = yaml.safe_load(f)

    # Step 5: Interactive ROI drawing
    roi_config = draw_rois_interactively(gm_image, roi_config, pipeline_cfg)

    # Step 6: Verify
    verify_rois(gm_image, roi_config)

    # Step 7: Confirm and save
    confirm = input("\nSave ROI config? [Y/n]: ").strip().lower()
    if confirm in ("", "y", "yes"):
        roi_config["configured"] = True
        with open(ROI_CONFIG_PATH, "w") as f:
            yaml.dump(roi_config, f, default_flow_style=False, sort_keys=False)
        print(f"\n[OK] ROI config saved to: {ROI_CONFIG_PATH}")
        print("[OK] You can now run:  python scripts/run_preprocessing.py")
    else:
        print("[ABORTED] No changes saved.")


if __name__ == "__main__":
    main()
