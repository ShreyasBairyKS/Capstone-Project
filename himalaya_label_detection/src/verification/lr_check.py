"""
lr_check.py
───────────
Left-Right symmetry check using Structural Similarity Index (SSIM).

The Himalaya cream label is printed as a two-up strip — both halves are
visible side-by-side in the camera image. A good label has near-identical
left and right halves. Defects such as a missing print on one side, a tear
on one half, or a misregistered two-up print cause SSIM to drop.

Why SSIM over pixel MAE:
  SSIM compares luminance, contrast, AND structure simultaneously.
  It is tolerant of minor tonal drift between print halves (acceptable
  process variation) but sensitive to structural differences (missing text,
  ink blobs, tears).
"""

import cv2
import numpy as np
from dataclasses import dataclass
from skimage.metrics import structural_similarity as ssim


@dataclass
class LRCheckResult:
    score:     float           # SSIM score 0.0–1.0 (higher = more similar)
    passed:    bool            # True if score >= threshold
    diff_map:  np.ndarray      # Per-pixel difference heatmap (BGR uint8)
    threshold: float


def check_left_right(
    aligned_image: np.ndarray,
    threshold: float = 0.85,
) -> LRCheckResult:
    """
    Compare the left and right halves of an aligned label image.

    Args:
        aligned_image: Full aligned label image (BGR or grayscale).
        threshold:     Minimum SSIM score to pass.

    Returns:
        LRCheckResult with score, pass/fail flag, and a colour difference heatmap.
    """
    h, w = aligned_image.shape[:2]

    gray = (cv2.cvtColor(aligned_image, cv2.COLOR_BGR2GRAY)
            if aligned_image.ndim == 3 else aligned_image.copy())

    left  = gray[:, : w // 2]
    right = gray[:, w // 2 : w // 2 * 2]   # same width if w is odd

    # Mirror right half so both are in the same orientation
    right_flipped = np.fliplr(right)

    # Guard against 1-pixel shape mismatch on odd-width images
    if left.shape != right_flipped.shape:
        right_flipped = cv2.resize(
            right_flipped, (left.shape[1], left.shape[0]),
            interpolation=cv2.INTER_AREA,
        )

    score, diff = ssim(left, right_flipped, full=True, data_range=255)

    # Invert diff (ssim diff is 1=identical → 0=different), then map to colour
    diff_uint8 = np.clip((1.0 - diff) * 255, 0, 255).astype(np.uint8)
    diff_color = cv2.applyColorMap(diff_uint8, cv2.COLORMAP_JET)

    return LRCheckResult(
        score=float(score),
        passed=float(score) >= threshold,
        diff_map=diff_color,
        threshold=threshold,
    )
