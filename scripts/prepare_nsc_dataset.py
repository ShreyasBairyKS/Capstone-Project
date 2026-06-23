"""
scripts/prepare_nsc_dataset.py
===============================
Data preparation pipeline for the NSC cylindrical tube rotation-unwrap
anomaly detection module.

Converts raw 1504×8000 BMP panoramic strips into:
  1. Quality-gate filtered captures
  2. Geometrically registered strips (fiducial-aligned)
  3. Overlapping 384×384 patches from the label band (for PatchCore)
  4. Golden profile (per-pixel mean ± std from good-only registered strips)

Usage:
    # Full preparation
    python scripts/prepare_nsc_dataset.py --data-root dataset/NSC

    # Dry run on first N images
    python scripts/prepare_nsc_dataset.py --data-root dataset/NSC --max-images 5

    # Skip quality gate (process all images regardless)
    python scripts/prepare_nsc_dataset.py --data-root dataset/NSC --skip-gate

    # Custom output directory
    python scripts/prepare_nsc_dataset.py --data-root dataset/NSC --output dataset/NSC_prepared
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
import yaml


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

IMG_EXTENSIONS = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}

DEFAULT_CONFIG = Path("configs/sku_profiles/nsc_tube_unwrap.yaml")
DEFAULT_OUTPUT = Path("dataset/NSC_prepared")


# ─────────────────────────────────────────────────────────────────────────────
# Configuration loader
# ─────────────────────────────────────────────────────────────────────────────

def load_config(config_path: Path) -> dict:
    """Load SKU profile YAML configuration."""
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────────────────────────────────────────
# Stage 0: Quality Gate
# ─────────────────────────────────────────────────────────────────────────────

def detect_skin_occlusion(
    image_bgr: np.ndarray,
    hsv_lower: list[int],
    hsv_upper: list[int],
    hsv_lower_alt: list[int] | None = None,
    hsv_upper_alt: list[int] | None = None,
    min_area: int = 5000,
) -> tuple[float, np.ndarray]:
    """
    Detect skin-tone occlusion via HSV thresholding.

    Returns:
        coverage_ratio: fraction of image width covered by skin-tone blobs
        mask: binary mask of detected skin regions
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(hsv_lower), np.array(hsv_upper))

    # Add secondary hue range (reddish tones wrapping around H=180)
    if hsv_lower_alt is not None and hsv_upper_alt is not None:
        mask_alt = cv2.inRange(hsv, np.array(hsv_lower_alt), np.array(hsv_upper_alt))
        mask = cv2.bitwise_or(mask, mask_alt)

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Find contiguous regions and filter by area
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_mask = np.zeros_like(mask)
    for cnt in contours:
        if cv2.contourArea(cnt) >= min_area:
            cv2.drawContours(filtered_mask, [cnt], -1, 255, -1)

    # Compute coverage along the width (rotation) axis
    col_coverage = np.any(filtered_mask > 0, axis=0).astype(np.float32)
    coverage_ratio = float(np.mean(col_coverage))

    return coverage_ratio, filtered_mask


def detect_glare(
    image_bgr: np.ndarray,
    value_threshold: int = 250,
    saturation_max: int = 30,
    critical_zone_rows: tuple[int, int] | None = None,
    min_area: int = 500,
) -> tuple[float, np.ndarray]:
    """
    Detect specular glare / blown-out highlights.

    Glare = high value + low saturation (washed-out white).

    Returns:
        area_ratio: fraction of critical zone covered by glare
        mask: binary mask of detected glare regions
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    if critical_zone_rows is not None:
        y_start, y_end = critical_zone_rows
        hsv_zone = hsv[y_start:y_end, :, :]
    else:
        hsv_zone = hsv
        y_start = 0

    # High value + low saturation = blown-out white
    value_mask = hsv_zone[:, :, 2] >= value_threshold
    sat_mask = hsv_zone[:, :, 1] <= saturation_max
    glare_mask = (value_mask & sat_mask).astype(np.uint8) * 255

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    glare_mask = cv2.morphologyEx(glare_mask, cv2.MORPH_CLOSE, kernel)

    # Filter small regions
    contours, _ = cv2.findContours(glare_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_mask = np.zeros_like(glare_mask)
    for cnt in contours:
        if cv2.contourArea(cnt) >= min_area:
            cv2.drawContours(filtered_mask, [cnt], -1, 255, -1)

    area_ratio = float(np.count_nonzero(filtered_mask)) / max(filtered_mask.size, 1)

    # Place back into full-image coordinates
    full_mask = np.zeros((image_bgr.shape[0], image_bgr.shape[1]), dtype=np.uint8)
    full_mask[y_start : y_start + filtered_mask.shape[0], :] = filtered_mask

    return area_ratio, full_mask


def run_quality_gate(
    image_bgr: np.ndarray,
    config: dict,
) -> dict:
    """
    Run Stage 0 quality gate on a single image.

    Returns dict with keys:
        status: "PASS" | "RECAPTURE"
        reason: str
        skin_coverage: float
        glare_coverage: float
        skin_mask: np.ndarray (optional)
        glare_mask: np.ndarray (optional)
    """
    qg = config["quality_gate"]

    # Occlusion detection
    skin_coverage, skin_mask = detect_skin_occlusion(
        image_bgr,
        hsv_lower=qg["skin_hsv_lower"],
        hsv_upper=qg["skin_hsv_upper"],
        hsv_lower_alt=qg.get("skin_hsv_lower_alt"),
        hsv_upper_alt=qg.get("skin_hsv_upper_alt"),
        min_area=qg.get("occlusion_min_area", 5000),
    )

    if skin_coverage > qg["occlusion_threshold"]:
        return {
            "status": "RECAPTURE",
            "reason": f"Skin occlusion detected: {skin_coverage:.1%} coverage "
                      f"(threshold: {qg['occlusion_threshold']:.1%})",
            "skin_coverage": skin_coverage,
            "glare_coverage": 0.0,
        }

    # Glare detection
    critical_rows = qg.get("glare_critical_zone_rows")
    critical_zone = tuple(critical_rows) if critical_rows else None
    glare_coverage, glare_mask = detect_glare(
        image_bgr,
        value_threshold=qg["glare_value_threshold"],
        saturation_max=qg.get("glare_saturation_max", 30),
        critical_zone_rows=critical_zone,
        min_area=500,
    )

    if glare_coverage > qg["glare_area_threshold"]:
        return {
            "status": "RECAPTURE",
            "reason": f"Glare detected: {glare_coverage:.1%} of critical zone "
                      f"(threshold: {qg['glare_area_threshold']:.1%})",
            "skin_coverage": skin_coverage,
            "glare_coverage": glare_coverage,
        }

    return {
        "status": "PASS",
        "reason": "Quality gate passed",
        "skin_coverage": skin_coverage,
        "glare_coverage": glare_coverage,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1: Geometric Registration
# ─────────────────────────────────────────────────────────────────────────────

def extract_fiducial_template(
    reference_image: np.ndarray,
    zone_rows: dict,
    save_path: Path | None = None,
) -> np.ndarray:
    """
    Auto-extract a fiducial template from a reference good image.

    Strategy: find the highest-contrast, most feature-rich region in the
    label band using Laplacian variance as a sharpness/detail metric.
    """
    label_start, label_end = zone_rows["label_band"]
    label_band = reference_image[label_start:label_end, :, :]

    h, w = label_band.shape[:2]
    template_h, template_w = min(200, h // 3), min(300, w // 20)

    best_score = -1.0
    best_crop = None
    best_pos = (0, 0)

    stride = 100
    gray_band = cv2.cvtColor(label_band, cv2.COLOR_BGR2GRAY)

    for y in range(0, h - template_h, stride):
        for x in range(0, w - template_w, stride):
            patch = gray_band[y : y + template_h, x : x + template_w]
            # Laplacian variance = sharpness / feature richness
            score = cv2.Laplacian(patch, cv2.CV_64F).var()
            if score > best_score:
                best_score = score
                best_crop = label_band[y : y + template_h, x : x + template_w].copy()
                best_pos = (x, y + label_start)

    if best_crop is None:
        raise RuntimeError("Could not extract fiducial template -- label band may be blank")

    print(f"  [FIDUCIAL] Extracted template at ({best_pos[0]}, {best_pos[1]}), "
          f"size={best_crop.shape[1]}x{best_crop.shape[0]}, score={best_score:.1f}")

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_path), best_crop)
        print(f"  [FIDUCIAL] Saved template -> {save_path}")

    return best_crop


def find_fiducial_offset(
    image_bgr: np.ndarray,
    template: np.ndarray,
    zone_rows: dict,
    method: int = cv2.TM_CCOEFF_NORMED,
    threshold: float = 0.4,
) -> tuple[int, float]:
    """
    Find the rotation-start offset by template matching the fiducial.

    Returns:
        x_offset: horizontal offset of best match
        match_score: template match correlation score
    """
    label_start, label_end = zone_rows["label_band"]
    label_band = image_bgr[label_start:label_end, :, :]

    result = cv2.matchTemplate(label_band, template, method)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)

    if max_val < threshold:
        print(f"  [WARN] Fiducial match score {max_val:.3f} < threshold {threshold}")

    return max_loc[0], float(max_val)


def measure_symmetry_offset(
    image_bgr: np.ndarray,
    zone_rows: dict,
    expected_offset: int = 4000,
    num_bands: int = 10,
    band_height: int = 100,
) -> tuple[int, float, dict]:
    """
    Measure the 2-fold rotational symmetry offset via circular cross-correlation
    on multiple horizontal bands within the label zone.

    Returns:
        measured_offset: consensus offset in pixels
        confidence: fraction of bands that agree with consensus
        details: per-band correlation info
    """
    label_start, label_end = zone_rows["label_band"]
    label_band = cv2.cvtColor(
        image_bgr[label_start:label_end, :, :], cv2.COLOR_BGR2GRAY
    ).astype(np.float64)

    h, w = label_band.shape
    band_positions = np.linspace(0, h - band_height, num_bands, dtype=int)

    offsets = []
    band_details = []

    for i, y in enumerate(band_positions):
        band = label_band[y : y + band_height, :]
        # Average across band height to get a 1D signal
        signal = np.mean(band, axis=0)
        signal -= np.mean(signal)

        # Circular cross-correlation via FFT
        fft_signal = np.fft.fft(signal)
        power = fft_signal * np.conj(fft_signal)
        xcorr = np.fft.ifft(power).real
        xcorr /= max(xcorr[0], 1e-10)  # Normalize

        # Search for peak near expected offset
        search_start = max(0, expected_offset - 200)
        search_end = min(w, expected_offset + 200)
        search_region = xcorr[search_start:search_end]

        if len(search_region) == 0:
            continue

        local_peak = int(np.argmax(search_region)) + search_start
        peak_corr = float(xcorr[local_peak])

        offsets.append(local_peak)
        band_details.append({
            "band_index": i,
            "band_y": int(y),
            "measured_offset": local_peak,
            "correlation": round(peak_corr, 4),
        })

    if not offsets:
        return expected_offset, 0.0, {"bands": band_details}

    # Consensus: take median offset, count how many bands agree within tolerance
    median_offset = int(np.median(offsets))
    tolerance = 20
    agreeing = sum(1 for o in offsets if abs(o - median_offset) <= tolerance)
    confidence = agreeing / len(offsets)

    return median_offset, confidence, {"bands": band_details}


def register_strip(
    image_bgr: np.ndarray,
    fiducial_offset: int,
    image_width: int = 8000,
) -> np.ndarray:
    """
    Roll the image horizontally so the fiducial starts at column 0.
    This aligns all captures to the same rotational starting point.
    """
    return np.roll(image_bgr, -fiducial_offset, axis=1)


# ─────────────────────────────────────────────────────────────────────────────
# Patch Extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_patches(
    registered_strip: np.ndarray,
    zone_rows: dict,
    patch_size: int = 384,
    stride: int = 192,
) -> list[np.ndarray]:
    """
    Extract overlapping patches from the label band of a registered strip.

    Returns list of (patch_size × patch_size × 3) BGR patches.
    """
    label_start, label_end = zone_rows["label_band"]
    label_band = registered_strip[label_start:label_end, :, :]
    h, w = label_band.shape[:2]

    patches = []
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            patch = label_band[y : y + patch_size, x : x + patch_size]
            if patch.shape[0] == patch_size and patch.shape[1] == patch_size:
                patches.append(patch)

    return patches


# ─────────────────────────────────────────────────────────────────────────────
# Golden Profile
# ─────────────────────────────────────────────────────────────────────────────

class GoldenProfileAccumulator:
    """
    Online accumulator for per-pixel mean and variance using Welford's algorithm.
    Memory-efficient: never holds all images in memory simultaneously.
    """

    def __init__(self) -> None:
        self.count = 0
        self.mean: np.ndarray | None = None
        self.m2: np.ndarray | None = None

    def update(self, image: np.ndarray) -> None:
        """Add a registered strip (float64 expected, or will be converted)."""
        img = image.astype(np.float64)
        self.count += 1

        if self.mean is None:
            self.mean = img.copy()
            self.m2 = np.zeros_like(img)
        else:
            delta = img - self.mean
            self.mean += delta / self.count
            delta2 = img - self.mean
            self.m2 += delta * delta2

    def get_mean(self) -> np.ndarray:
        if self.mean is None:
            raise RuntimeError("No images accumulated")
        return self.mean

    def get_std(self) -> np.ndarray:
        if self.count < 2:
            raise RuntimeError(f"Need >=2 samples for std, have {self.count}")
        return np.sqrt(self.m2 / (self.count - 1))

    def save(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        np.save(output_dir / "golden_mean.npy", self.get_mean())
        np.save(output_dir / "golden_std.npy", self.get_std())
        print(f"  [GOLDEN] Saved mean/std from {self.count} good samples -> {output_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# Image I/O helpers
# ─────────────────────────────────────────────────────────────────────────────

def list_images(folder: Path) -> list[Path]:
    """List image files sorted by numeric stem if possible."""
    images = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTENSIONS]

    def sort_key(p: Path):
        try:
            return int(p.stem)
        except ValueError:
            return p.stem

    return sorted(images, key=sort_key)


def load_image(path: Path) -> np.ndarray:
    """Load a large BMP image. Returns BGR numpy array."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"Failed to load image: {path}")
    return img


# ─────────────────────────────────────────────────────────────────────────────
# CLI & Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare NSC cylindrical tube dataset for anomaly detection training."
    )
    parser.add_argument(
        "--data-root", type=Path, default=Path("dataset/NSC"),
        help="Root directory containing 'NSC GOOD IMAGES' and 'NSC BAD IMAGES' subfolders",
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT,
        help=f"Output directory for prepared data (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG,
        help=f"SKU profile YAML config (default: {DEFAULT_CONFIG})",
    )
    parser.add_argument("--max-images", type=int, default=0,
                        help="Process only first N images per class (0=all)")
    parser.add_argument("--skip-gate", action="store_true",
                        help="Skip quality gate (process all images)")
    parser.add_argument("--skip-patches", action="store_true",
                        help="Skip patch extraction")
    parser.add_argument("--skip-golden", action="store_true",
                        help="Skip golden profile computation")
    parser.add_argument("--val-ratio", type=float, default=0.20,
                        help="Fraction of good images for validation (default: 0.20)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patch-size", type=int, default=384)
    parser.add_argument("--stride", type=int, default=192)
    parser.add_argument("--dry-run", action="store_true",
                        help="Print actions without writing files")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    data_root = args.data_root.resolve()
    output_dir = args.output.resolve()

    good_dir = data_root / "NSC GOOD IMAGES"
    bad_dir = data_root / "NSC BAD IMAGES"

    if not good_dir.exists():
        raise FileNotFoundError(f"Good images directory not found: {good_dir}")
    if not bad_dir.exists():
        raise FileNotFoundError(f"Bad images directory not found: {bad_dir}")

    good_images = list_images(good_dir)
    bad_images = list_images(bad_dir)

    if args.max_images > 0:
        good_images = good_images[: args.max_images]
        bad_images = bad_images[: args.max_images]

    print(f"\n{'=' * 70}")
    print("  NSC Rotation-Unwrap Dataset Preparation")
    print(f"{'=' * 70}")
    print(f"  Good images : {len(good_images)}")
    print(f"  Bad images  : {len(bad_images)}")
    print(f"  Output      : {output_dir}")
    print(f"  Patch size  : {args.patch_size}x{args.patch_size}")
    print(f"  Stride      : {args.stride}")
    print(f"  Val ratio   : {args.val_ratio}")
    print(f"  Dry run     : {args.dry_run}")
    print(f"{'=' * 70}\n")

    if args.dry_run:
        print("[DRY RUN] Would process the above images. Exiting.")
        return

    # Create output directories
    reg = config["registration"]
    zone_rows = reg["zone_rows"]
    patchcore_cfg = config["patchcore"]

    out_registered_good = output_dir / "registered" / "good"
    out_registered_bad = output_dir / "registered" / "bad"
    out_patches_good_train = output_dir / "patches" / "good_train"
    out_patches_good_val = output_dir / "patches" / "good_val"
    out_patches_bad_val = output_dir / "patches" / "bad_val"
    out_golden = output_dir / "golden_profile"

    for d in [out_registered_good, out_registered_bad,
              out_patches_good_train, out_patches_good_val,
              out_patches_bad_val, out_golden]:
        d.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Extract fiducial template from first good image ──
    print("\n[STEP 1] Extracting fiducial template from reference image...")
    ref_image = load_image(good_images[0])
    template_path = Path(reg.get("fiducial_template", "configs/templates/nsc_logo_template.png"))
    if template_path.exists():
        print(f"  [FIDUCIAL] Using existing template: {template_path}")
        fiducial_template = cv2.imread(str(template_path), cv2.IMREAD_COLOR)
    else:
        fiducial_template = extract_fiducial_template(
            ref_image, zone_rows, save_path=template_path
        )
    del ref_image  # Free ~36MB

    # ── Step 2: Train/val split for good images ──
    print(f"\n[STEP 2] Splitting good images (val_ratio={args.val_ratio})...")
    rng = random.Random(args.seed)
    good_indices = list(range(len(good_images)))
    rng.shuffle(good_indices)
    n_val = max(1, int(len(good_images) * args.val_ratio))
    val_indices = set(good_indices[:n_val])
    train_indices = set(good_indices[n_val:])
    print(f"  Train: {len(train_indices)} | Val: {len(val_indices)}")

    # ── Step 3: Process all images ──
    quality_gate_log = []
    registration_log = []
    golden_accumulator = GoldenProfileAccumulator()

    total_train_patches = 0
    total_val_patches = 0
    total_bad_patches = 0

    def process_images(
        image_paths: list[Path],
        label: str,
        is_good: bool,
        good_split_indices: set[int] | None = None,
    ):
        nonlocal total_train_patches, total_val_patches, total_bad_patches

        print(f"\n[STEP 3] Processing {label} images ({len(image_paths)} total)...")

        for i, img_path in enumerate(image_paths):
            t0 = time.time()
            print(f"  [{i + 1}/{len(image_paths)}] {img_path.name}...", end=" ", flush=True)

            img = load_image(img_path)

            # Quality gate
            if not args.skip_gate:
                gate_result = run_quality_gate(img, config)
                gate_entry = {
                    "image": img_path.name,
                    "label": label,
                    "status": gate_result["status"],
                    "reason": gate_result["reason"],
                    "skin_coverage": round(gate_result["skin_coverage"], 4),
                    "glare_coverage": round(gate_result["glare_coverage"], 4),
                }
                quality_gate_log.append(gate_entry)

                if gate_result["status"] == "RECAPTURE":
                    print(f"RECAPTURE ({gate_result['reason']})")
                    del img
                    continue

            # Registration: find fiducial offset
            x_offset, match_score = find_fiducial_offset(
                img, fiducial_template, zone_rows,
                threshold=reg.get("template_match_threshold", 0.4),
            )

            # Measure symmetry offset
            sym_offset, sym_confidence, sym_details = measure_symmetry_offset(
                img, zone_rows,
                expected_offset=reg["expected_symmetry_offset"],
                num_bands=reg.get("num_correlation_bands", 10),
                band_height=reg.get("correlation_band_height", 100),
            )

            reg_entry = {
                "image": img_path.name,
                "label": label,
                "fiducial_x_offset": x_offset,
                "fiducial_match_score": round(match_score, 4),
                "symmetry_offset": sym_offset,
                "symmetry_confidence": round(sym_confidence, 4),
                "offset_deviation": abs(sym_offset - reg["expected_symmetry_offset"]),
            }
            registration_log.append(reg_entry)

            # Register the strip
            registered = register_strip(img, x_offset, image_width=img.shape[1])

            # Save registered strip
            if is_good:
                out_reg_dir = out_registered_good
            else:
                out_reg_dir = out_registered_bad
            cv2.imwrite(
                str(out_reg_dir / f"{img_path.stem}.png"),
                registered,
                [cv2.IMWRITE_PNG_COMPRESSION, 3],
            )

            # Golden profile accumulation (good training images only)
            if is_good and not args.skip_golden and good_split_indices is not None:
                if i in train_indices:
                    golden_accumulator.update(registered)

            # Patch extraction
            if not args.skip_patches:
                patches = extract_patches(
                    registered, zone_rows,
                    patch_size=args.patch_size,
                    stride=args.stride,
                )

                if is_good and good_split_indices is not None:
                    if i in val_indices:
                        patch_dir = out_patches_good_val
                        total_val_patches += len(patches)
                    else:
                        patch_dir = out_patches_good_train
                        total_train_patches += len(patches)
                else:
                    patch_dir = out_patches_bad_val
                    total_bad_patches += len(patches)

                for pi, patch in enumerate(patches):
                    patch_name = f"{img_path.stem}_p{pi:04d}.png"
                    cv2.imwrite(str(patch_dir / patch_name), patch)

            elapsed = time.time() - t0
            print(
                f"OK (reg_offset={x_offset}, sym={sym_offset}, "
                f"conf={sym_confidence:.2f}, "
                f"patches={len(patches) if not args.skip_patches else 'skip'}, "
                f"{elapsed:.1f}s)"
            )

            del img, registered  # Free memory

    # Process good images
    process_images(good_images, "good", is_good=True, good_split_indices=train_indices)

    # Process bad images
    process_images(bad_images, "bad", is_good=False)

    # ── Step 4: Save golden profile ──
    if not args.skip_golden:
        print(f"\n[STEP 4] Saving golden profile ({golden_accumulator.count} good samples)...")
        if golden_accumulator.count >= 2:
            golden_accumulator.save(out_golden)
        else:
            print("  [WARN] Not enough good samples for golden profile (need >=2)")

    # ── Step 5: Save logs ──
    print("\n[STEP 5] Saving logs...")
    with open(output_dir / "quality_gate_log.json", "w", encoding="utf-8") as f:
        json.dump(quality_gate_log, f, indent=2)
    with open(output_dir / "registration_log.json", "w", encoding="utf-8") as f:
        json.dump(registration_log, f, indent=2)

    # Summary
    gate_passed = sum(1 for e in quality_gate_log if e["status"] == "PASS")
    gate_recapture = sum(1 for e in quality_gate_log if e["status"] == "RECAPTURE")

    print(f"\n{'=' * 70}")
    print("  Preparation Complete")
    print(f"{'=' * 70}")
    print(f"  Quality gate: {gate_passed} PASS, {gate_recapture} RECAPTURE")
    print(f"  Registered strips: good={len(list(out_registered_good.glob('*.png')))}, "
          f"bad={len(list(out_registered_bad.glob('*.png')))}")
    print(f"  Patches (train good): {total_train_patches}")
    print(f"  Patches (val good)  : {total_val_patches}")
    print(f"  Patches (val bad)   : {total_bad_patches}")
    if not args.skip_golden:
        print(f"  Golden profile      : {golden_accumulator.count} samples")
    print(f"  Output directory    : {output_dir}")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
