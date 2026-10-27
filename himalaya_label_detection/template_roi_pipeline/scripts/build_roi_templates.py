r"""
build_roi_templates.py
-----------------------------------------------------------------------------
Create the four full-resolution ROI templates used by ROITemplateMatcher.

The repo does not ship the patch_roi*.png anchors referenced in
roi_coord_config.json, so this script rebuilds them from ONE clean reference
scan:

  1. Each docs/roiN_sample.png (a downscaled example of the ROI) is resized to
     the ROI's reference size (w x h from roi_coord_config.json) and matched
     against the reference scan to find where that ROI sits.
     Or pass explicit positions with --coords ROI_1=x,y ROI_2=x,y ...
  2. The full-resolution grayscale crop at that position is saved to
     template_roi_pipeline/config/roi_templates/roi_N_template.png.

ROI sizes come from himalaya_label_detection/config/roi_coord_config.json and the
sample images from himalaya_label_detection/docs (override with --coord-config-dir /
--samples-dir).

Usage:
    python himalaya_label_detection/template_roi_pipeline/scripts/build_roi_templates.py \
        --reference "../NSC/NSC GOOD IMAGES/1.bmp"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

PKG_ROOT = Path(__file__).resolve().parents[1]          # himalaya_label_detection/template_roi_pipeline
LABEL_ROOT = PKG_ROOT.parent                            # himalaya_label_detection
sys.path.insert(0, str(LABEL_ROOT))

from template_roi_pipeline.roi_template_matcher import ROI_NAMES, load_gray, load_roi_sizes  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--reference", type=Path, required=True, help="clean full-size reference scan (BMP)")
    p.add_argument("--config-dir", type=Path, default=PKG_ROOT / "config",
                   help="where roi_templates/ is written")
    p.add_argument("--coord-config-dir", type=Path, default=LABEL_ROOT / "config",
                   help="folder holding roi_coord_config.json (ROI sizes)")
    p.add_argument("--samples-dir", type=Path, default=LABEL_ROOT / "docs",
                   help="folder with roi1_sample.png .. roi4_sample.png (used to locate each ROI)")
    p.add_argument("--coords", nargs="*", default=[],
                   help="manual override, e.g. ROI_1=256,1264 (x,y of the top-left corner)")
    return p.parse_args()


def locate_from_sample(gray, sample_path: Path, w: int, h: int, scale: int = 4):
    sample = cv2.imread(str(sample_path), cv2.IMREAD_GRAYSCALE)
    if sample is None:
        raise SystemExit(f"Cannot read sample: {sample_path}")
    native = cv2.resize(sample, (w, h), interpolation=cv2.INTER_AREA)
    small_img = cv2.resize(gray, None, fx=1 / scale, fy=1 / scale, interpolation=cv2.INTER_AREA)
    small_t = cv2.resize(native, None, fx=1 / scale, fy=1 / scale, interpolation=cv2.INTER_AREA)
    res = cv2.matchTemplate(small_img, small_t, cv2.TM_CCOEFF_NORMED)
    _, score, _, (cx, cy) = cv2.minMaxLoc(res)
    gx, gy = cx * scale, cy * scale
    r = 2 * scale
    x0, y0 = max(0, gx - r), max(0, gy - r)
    window = gray[y0:gy + h + r, x0:gx + w + r]
    fine = cv2.matchTemplate(window, native, cv2.TM_CCOEFF_NORMED)
    _, score, _, (fx, fy) = cv2.minMaxLoc(fine)
    return x0 + fx, y0 + fy, float(score)


def main() -> int:
    args = parse_args()
    gray = load_gray(args.reference)
    sizes = load_roi_sizes(args.coord_config_dir)
    manual = {}
    for item in args.coords:
        name, xy = item.split("=")
        manual[name] = tuple(int(v) for v in xy.split(","))

    out_dir = args.config_dir / "roi_templates"
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, name in enumerate(ROI_NAMES, start=1):
        w, h = sizes[name]
        if name in manual:
            x, y = manual[name]
            score = float("nan")
        else:
            x, y, score = locate_from_sample(gray, args.samples_dir / f"roi{i}_sample.png", w, h)
        crop = gray[y:y + h, x:x + w]
        if crop.shape != (h, w):
            raise SystemExit(f"{name}: crop at ({x},{y}) is {crop.shape[::-1]}, expected {(w, h)}")
        path = out_dir / f"{name.lower()}_template.png"
        cv2.imwrite(str(path), crop)
        print(f"{name}: x={x} y={y} size={w}x{h} sample-match={score:.3f} -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
