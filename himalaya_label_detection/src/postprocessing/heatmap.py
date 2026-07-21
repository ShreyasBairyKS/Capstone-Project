"""
heatmap.py
──────────
Anomaly heatmap generation and defect crop extraction.

PatchCore / EfficientAD produce a per-pixel anomaly score map as a
by-product of inference. This module:
  1. Normalises the map and blends it onto the original image as a
     colour overlay so operators can immediately see WHERE the defect is.
  2. Locates the highest-anomaly region, expands it with a margin, and
     crops it for downstream segmentation (if configured).

Run only on FAIL images — saves ~20 ms per label on the majority (good) path.
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional, Tuple


@dataclass
class HeatmapResult:
    overlay:       np.ndarray                           # BGR image with heatmap blended in
    anomaly_crop:  Optional[np.ndarray]                 # Tight BGR crop around worst anomaly
    crop_box:      Optional[Tuple[int, int, int, int]]  # (x1, y1, x2, y2) in aligned image
    peak_score:    float                                # Max anomaly score in the map


def generate_heatmap(
    image: np.ndarray,
    anomaly_map: np.ndarray,
    alpha: float = 0.45,
    crop_margin: int = 20,
    crop_top_percentile: float = 95.0,
) -> HeatmapResult:
    """
    Overlay the anomaly map on the image and extract the worst-anomaly crop.

    Args:
        image:               Aligned BGR label image.
        anomaly_map:         Per-pixel anomaly scores (float32, any scale).
        alpha:               Heatmap opacity (0 = image only, 1 = heatmap only).
        crop_margin:         Extra pixels added around the defect bounding box.
        crop_top_percentile: Percentile cutoff to define "hot" pixels
                             (95 = flag the top 5% of anomaly pixels).

    Returns:
        HeatmapResult with overlay, optional crop, box coordinates, and peak score.
    """
    h, w = image.shape[:2]

    # Normalise to 0–255
    norm_map = cv2.normalize(
        anomaly_map.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX
    ).astype(np.uint8)

    # Resize to match image if anomaly map is at lower resolution
    if norm_map.shape[:2] != (h, w):
        norm_map = cv2.resize(norm_map, (w, h), interpolation=cv2.INTER_LINEAR)

    # Jet colourmap: blue=normal → green → red=anomaly
    colormap = cv2.applyColorMap(norm_map, cv2.COLORMAP_JET)
    overlay  = cv2.addWeighted(image, 1.0 - alpha, colormap, alpha, 0)

    # Find bounding box of hot pixels
    threshold_val = np.percentile(norm_map, crop_top_percentile)
    hot_mask      = (norm_map >= threshold_val).astype(np.uint8)

    crop     = None
    crop_box = None
    ys, xs   = np.where(hot_mask > 0)

    if len(xs) > 0:
        x1 = max(0, int(xs.min()) - crop_margin)
        y1 = max(0, int(ys.min()) - crop_margin)
        x2 = min(w, int(xs.max()) + crop_margin)
        y2 = min(h, int(ys.max()) + crop_margin)

        crop     = image[y1:y2, x1:x2].copy()
        crop_box = (x1, y1, x2, y2)

        # Draw bounding box on overlay
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 255), 2)

    return HeatmapResult(
        overlay=overlay,
        anomaly_crop=crop,
        crop_box=crop_box,
        peak_score=float(anomaly_map.max()),
    )


def combine_roi_heatmaps(
    image: np.ndarray,
    roi_anomaly_maps: Dict[str, np.ndarray],
    roi_configs: dict,
) -> np.ndarray:
    """
    Paste per-ROI anomaly maps back into a single full-label anomaly map.

    Args:
        image:            Full aligned BGR label image (used for shape only).
        roi_anomaly_maps: Dict of ROI name → per-ROI anomaly map (float32).
        roi_configs:      Dict of ROI name → ROIConfig (with .x .y .w .h).

    Returns:
        Full-resolution combined anomaly map (float32), same H×W as image.
    """
    h, w    = image.shape[:2]
    combined = np.zeros((h, w), dtype=np.float32)

    for roi_name, amap in roi_anomaly_maps.items():
        if roi_name not in roi_configs:
            continue
        cfg = roi_configs[roi_name]

        resized = cv2.resize(
            amap.astype(np.float32),
            (cfg.w, cfg.h),
            interpolation=cv2.INTER_LINEAR,
        )

        y1 = cfg.y
        y2 = min(h, cfg.y + cfg.h)
        x1 = cfg.x
        x2 = min(w, cfg.x + cfg.w)

        combined[y1:y2, x1:x2] = np.maximum(
            combined[y1:y2, x1:x2],
            resized[: y2 - y1, : x2 - x1],
        )

    return combined
