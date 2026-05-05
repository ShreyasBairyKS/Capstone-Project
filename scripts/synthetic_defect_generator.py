"""
scripts/synthetic_defect_generator.py
=====================================
Generate synthetic defective cap images from good cap images.

Solves the classification dataset problem when you only have good caps.

Defect types generated:
    1. Dent/deformation   — warps a region of the cap
    2. Scratch            — draws thin lines across cap surface
    3. Misalignment       — shifts/rotates the cap off-center
    4. Discoloration      — adds stains, dark spots, color patches
    5. Crack              — draws jagged crack lines
    6. Partial missing    — blacks out a section (broken cap)

Usage:
    python scripts/synthetic_defect_generator.py \
        --input  data/caps/good_cap \
        --output data/caps/defective_cap \
        --variants 5 \
        --seed 42
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


# ─────────────────────────────────────────────────────────────────────────────
# DEFECT GENERATORS
# ─────────────────────────────────────────────────────────────────────────────

def apply_dent(img: np.ndarray, rng: random.Random) -> np.ndarray:
    """Simulate a dent by locally warping a circular region."""
    h, w = img.shape[:2]
    result = img.copy()

    # Random center for dent
    cx = rng.randint(w // 4, 3 * w // 4)
    cy = rng.randint(h // 4, 3 * h // 4)
    radius = rng.randint(min(h, w) // 8, min(h, w) // 4)

    # Create displacement map for barrel/pincushion distortion
    map_x = np.zeros((h, w), dtype=np.float32)
    map_y = np.zeros((h, w), dtype=np.float32)

    for y in range(h):
        for x in range(w):
            dx, dy = x - cx, y - cy
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < radius and dist > 0:
                # Pincushion distortion inside the dent region
                factor = 1.0 + 0.4 * (1.0 - dist / radius) ** 2
                map_x[y, x] = cx + dx * factor
                map_y[y, x] = cy + dy * factor
            else:
                map_x[y, x] = x
                map_y[y, x] = y

    result = cv2.remap(result, map_x, map_y, cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_REFLECT)

    # Add slight shadow near dent edge for realism
    mask = np.zeros((h, w), dtype=np.float32)
    cv2.circle(mask, (cx, cy), radius, 1.0, -1)
    mask = cv2.GaussianBlur(mask, (21, 21), 0)
    shadow = (mask * 30).astype(np.uint8)
    result = cv2.subtract(result, cv2.merge([shadow, shadow, shadow]))

    return result


def apply_scratch(img: np.ndarray, rng: random.Random) -> np.ndarray:
    """Draw realistic scratch lines across the cap surface."""
    h, w = img.shape[:2]
    result = img.copy()
    n_scratches = rng.randint(1, 4)

    for _ in range(n_scratches):
        # Start and end points
        x1 = rng.randint(0, w)
        y1 = rng.randint(0, h)
        angle = rng.uniform(0, 2 * np.pi)
        length = rng.randint(min(h, w) // 3, min(h, w))
        x2 = int(x1 + length * np.cos(angle))
        y2 = int(y1 + length * np.sin(angle))

        # Draw with slight curve using polylines
        n_points = rng.randint(3, 6)
        points = []
        for i in range(n_points):
            t = i / (n_points - 1)
            px = int(x1 + t * (x2 - x1) + rng.randint(-10, 10))
            py = int(y1 + t * (y2 - y1) + rng.randint(-10, 10))
            points.append([px, py])

        pts = np.array(points, dtype=np.int32)
        thickness = rng.randint(1, 3)

        # Scratch color: lighter than surface (metallic scratch)
        color = (
            rng.randint(180, 230),
            rng.randint(180, 230),
            rng.randint(180, 230),
        )
        cv2.polylines(result, [pts], False, color, thickness, cv2.LINE_AA)

    return result


def apply_misalignment(img: np.ndarray, rng: random.Random) -> np.ndarray:
    """Shift/rotate the cap to simulate misalignment on bottle."""
    h, w = img.shape[:2]
    cx, cy = w / 2.0, h / 2.0

    # Random rotation (tilted cap)
    angle = rng.uniform(8, 25) * rng.choice([-1, 1])
    # Random translation (off-center cap)
    tx = rng.uniform(-w * 0.1, w * 0.1)
    ty = rng.uniform(-h * 0.1, h * 0.1)

    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    M[0, 2] += tx
    M[1, 2] += ty

    result = cv2.warpAffine(img, M, (w, h),
                            borderMode=cv2.BORDER_REFLECT_101)
    return result


def apply_discoloration(img: np.ndarray, rng: random.Random) -> np.ndarray:
    """Add stains, dark spots, or discolored patches."""
    h, w = img.shape[:2]
    result = img.copy()
    n_spots = rng.randint(1, 5)

    for _ in range(n_spots):
        cx = rng.randint(w // 6, 5 * w // 6)
        cy = rng.randint(h // 6, 5 * h // 6)
        radius = rng.randint(min(h, w) // 10, min(h, w) // 4)

        # Create spot mask
        mask = np.zeros((h, w), dtype=np.float32)
        cv2.circle(mask, (cx, cy), radius, 1.0, -1)
        mask = cv2.GaussianBlur(mask, (31, 31), 0)

        # Random discoloration
        spot_type = rng.choice(["dark", "rust", "bleach", "stain"])
        if spot_type == "dark":
            overlay = np.zeros_like(result)
        elif spot_type == "rust":
            overlay = np.full_like(result, [30, 80, 160])  # brownish
        elif spot_type == "bleach":
            overlay = np.full_like(result, [220, 220, 200])
        else:  # stain
            overlay = np.full_like(result, [
                rng.randint(0, 100),
                rng.randint(50, 150),
                rng.randint(0, 100),
            ])

        mask_3ch = cv2.merge([mask, mask, mask])
        alpha = rng.uniform(0.3, 0.7)
        result = (result * (1 - mask_3ch * alpha) +
                  overlay * mask_3ch * alpha).astype(np.uint8)

    return result


def apply_crack(img: np.ndarray, rng: random.Random) -> np.ndarray:
    """Draw jagged crack lines simulating a cracked cap."""
    h, w = img.shape[:2]
    result = img.copy()

    # Start from a random edge point
    start_x = rng.randint(w // 4, 3 * w // 4)
    start_y = rng.choice([0, h - 1]) if rng.random() > 0.5 else rng.randint(0, h)

    points = [(start_x, start_y)]
    n_segments = rng.randint(8, 20)

    for _ in range(n_segments):
        px, py = points[-1]
        dx = rng.randint(-15, 15)
        dy = rng.randint(5, 20) if start_y == 0 else rng.randint(-20, -5)
        nx = max(0, min(w - 1, px + dx))
        ny = max(0, min(h - 1, py + dy))
        points.append((nx, ny))

        # Branch cracks
        if rng.random() < 0.3:
            bx = nx + rng.randint(-20, 20)
            by = ny + rng.randint(-10, 10)
            bx = max(0, min(w - 1, bx))
            by = max(0, min(h - 1, by))
            cv2.line(result, (nx, ny), (bx, by),
                     (rng.randint(20, 60),) * 3, 1, cv2.LINE_AA)

    pts = np.array(points, dtype=np.int32)
    cv2.polylines(result, [pts], False,
                  (rng.randint(30, 80),) * 3,
                  rng.randint(1, 3), cv2.LINE_AA)

    return result


def apply_partial_missing(img: np.ndarray, rng: random.Random) -> np.ndarray:
    """Black out / corrupt a section to simulate a broken/missing piece."""
    h, w = img.shape[:2]
    result = img.copy()

    # Random polygon mask for missing section
    n_points = rng.randint(3, 6)
    angle_start = rng.uniform(0, 2 * np.pi)
    angle_span = rng.uniform(np.pi / 4, np.pi)

    cx, cy = w // 2, h // 2
    r_outer = min(h, w) // 2

    points = [(cx, cy)]
    for i in range(n_points):
        a = angle_start + (i / (n_points - 1)) * angle_span
        r = rng.uniform(r_outer * 0.5, r_outer)
        px = int(cx + r * np.cos(a))
        py = int(cy + r * np.sin(a))
        points.append((px, py))

    pts = np.array(points, dtype=np.int32)

    # Fill with dark color (missing material)
    fill_color = (
        rng.randint(10, 40),
        rng.randint(10, 40),
        rng.randint(10, 40),
    )
    cv2.fillPoly(result, [pts], fill_color)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

DEFECT_FUNCTIONS = [
    ("dent", apply_dent),
    ("scratch", apply_scratch),
    ("misalign", apply_misalignment),
    ("discolor", apply_discoloration),
    ("crack", apply_crack),
    ("partial", apply_partial_missing),
]


def generate_defective_variants(
    img: np.ndarray,
    n_variants: int,
    rng: random.Random,
) -> list[tuple[np.ndarray, str]]:
    """Generate multiple defective variants from one good cap image."""
    results = []

    for i in range(n_variants):
        # Pick 1-2 defect types to combine
        n_defects = rng.choices([1, 2], weights=[0.6, 0.4])[0]
        chosen = rng.sample(DEFECT_FUNCTIONS, min(n_defects, len(DEFECT_FUNCTIONS)))

        aug = img.copy()
        defect_labels = []
        for label, fn in chosen:
            aug = fn(aug, rng)
            defect_labels.append(label)

        # Add slight noise/blur for realism
        if rng.random() < 0.3:
            noise = np.random.normal(0, 8, aug.shape).astype(np.int16)
            aug = np.clip(aug.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        if rng.random() < 0.2:
            aug = cv2.GaussianBlur(aug, (3, 3), 0)

        suffix = "_".join(defect_labels)
        results.append((aug, f"v{i}_{suffix}"))

    return results


def letterbox_resize(img: np.ndarray, size: int) -> np.ndarray:
    """Resize preserving aspect ratio with padding (no distortion)."""
    h, w = img.shape[:2]
    ratio = min(size / h, size / w)
    new_h, new_w = int(round(h * ratio)), int(round(w * ratio))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_top = (size - new_h) // 2
    pad_left = (size - new_w) // 2
    padded = cv2.copyMakeBorder(
        resized,
        pad_top, size - new_h - pad_top,
        pad_left, size - new_w - pad_left,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )
    return padded


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic defective cap images from good caps"
    )
    parser.add_argument("--input", required=True,
                        help="Path to folder of good cap images")
    parser.add_argument("--output", required=True,
                        help="Path to output folder for defective cap images")
    parser.add_argument("--variants", type=int, default=5,
                        help="Number of defective variants per good image (default: 5)")
    parser.add_argument("--resize", type=int, default=0,
                        help="Resize images to this size (letterbox). 0 = no resize")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--also-copy-good", action="store_true",
                        help="Also copy/resize good caps to a parallel good_cap folder")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    assert input_dir.exists(), f"Input folder not found: {input_dir}"

    output_dir.mkdir(parents=True, exist_ok=True)

    if args.also_copy_good:
        good_out = output_dir.parent / "good_cap"
        good_out.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    image_files = sorted(
        p for p in input_dir.glob("*.*")
        if p.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not image_files:
        print(f"No images found in {input_dir}")
        return

    print(f"\n{'='*55}")
    print("  Synthetic Defect Generator")
    print(f"{'='*55}")
    print(f"  Input       : {input_dir}")
    print(f"  Output      : {output_dir}")
    print(f"  Good images : {len(image_files)}")
    print(f"  Variants    : {args.variants} per image")
    print(f"  Expected    : ~{len(image_files) * args.variants} defective images")
    print(f"  Resize      : {'letterbox ' + str(args.resize) if args.resize else 'none'}")
    print()

    total_generated = 0
    for img_path in image_files:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  [SKIP] Cannot read: {img_path.name}")
            continue

        if args.resize > 0:
            img = letterbox_resize(img, args.resize)

        # Copy good image if requested
        if args.also_copy_good:
            out_good = good_out / img_path.name
            cv2.imwrite(str(out_good), img, [cv2.IMWRITE_JPEG_QUALITY, 95])

        # Generate defective variants
        variants = generate_defective_variants(img, args.variants, rng)
        for aug_img, label in variants:
            out_name = f"{img_path.stem}_{label}{img_path.suffix}"
            out_path = output_dir / out_name
            cv2.imwrite(str(out_path), aug_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            total_generated += 1

    print(f"  Generated {total_generated} defective cap images")
    print(f"  Saved to: {output_dir.resolve()}")
    print(f"\n  Dataset structure for classifier training:")
    print(f"    data/caps/")
    print(f"      good_cap/       ← your original good caps")
    print(f"      defective_cap/  ← generated by this script")
    print(f"\n  → Next: split into train/val with train_cap_classifier.py --auto-split\n")


if __name__ == "__main__":
    main()
