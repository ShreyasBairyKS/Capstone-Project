"""
bbox.py
───────
Bounding box extraction from per-pixel anomaly score maps.

EfficientAD and PatchCore both produce a float32 anomaly_map (H×W) where
higher values = more anomalous. This module converts that map into a list
of bounding boxes around detected anomaly clusters.

Pipeline:
    anomaly_map (float32 H×W)
        → normalize to 0-255
        → binary threshold (top-percentile pixels)
        → morphological close  (merge nearby blobs)
        → connected components
        → filter by min area
        → expand bbox by margin
        → return list of {x, y, w, h, score, area}

The coordinates returned are in the ROI crop's pixel space.
Use map_bboxes_to_full_image() to convert to full-aligned-image coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np


# ── Data types ────────────────────────────────────────────────────────────────

@dataclass
class BoundingBox:
    x: int          # left edge (pixels, ROI crop space)
    y: int          # top edge
    w: int          # width
    h: int          # height
    score: float    # mean anomaly score inside this bbox (0.0–1.0)
    area: int       # pixel area of the detected anomaly blob
    label: str = "anomaly"

    @property
    def x2(self) -> int:
        return self.x + self.w

    @property
    def y2(self) -> int:
        return self.y + self.h

    @property
    def center(self) -> Tuple[float, float]:
        return (self.x + self.w / 2, self.y + self.h / 2)

    def to_dict(self) -> dict:
        return {
            "x": self.x, "y": self.y,
            "w": self.w, "h": self.h,
            "x2": self.x2, "y2": self.y2,
            "score": round(self.score, 4),
            "area": self.area,
            "label": self.label,
        }


@dataclass
class ROIDetectionResult:
    roi_name: str
    anomaly_score: float          # scalar score for this ROI (max or mean)
    anomaly_map: Optional[np.ndarray]  # per-pixel float32 map (H×W)
    bounding_boxes: List[BoundingBox]  # detected anomaly regions
    is_anomalous: bool
    threshold_used: float

    def to_dict(self) -> dict:
        return {
            "roi_name":       self.roi_name,
            "anomaly_score":  round(self.anomaly_score, 4),
            "is_anomalous":   self.is_anomalous,
            "threshold_used": round(self.threshold_used, 4),
            "bounding_boxes": [b.to_dict() for b in self.bounding_boxes],
        }


# ── Core function ─────────────────────────────────────────────────────────────

def heatmap_to_bboxes(
    anomaly_map: np.ndarray,
    threshold_percentile: float = 92.0,
    min_area_px: int = 40,
    margin_px: int = 8,
    morph_kernel_size: int = 7,
) -> List[BoundingBox]:
    """
    Convert a per-pixel anomaly score map to bounding boxes.

    Args:
        anomaly_map:          float32 array shape (H, W). Higher = more anomalous.
                              May have any scale (will be normalised internally).
        threshold_percentile: Percentile cutoff. Pixels above this are "hot".
                              92 = top 8% of pixels are treated as anomalous.
                              Lower value → larger, more sensitive blobs.
                              Higher value → smaller, more precise blobs.
        min_area_px:          Minimum blob area in pixels. Blobs smaller than this
                              are discarded as noise.
        margin_px:            Extra pixels added around each bounding box.
        morph_kernel_size:    Morphological closing kernel size (px). Larger values
                              merge nearby blobs; smaller values keep them separate.

    Returns:
        List of BoundingBox, sorted by score descending (highest anomaly first).
        Empty list if no anomalous region is detected.
    """
    if anomaly_map is None or anomaly_map.size == 0:
        return []

    h_map, w_map = anomaly_map.shape[:2]

    # Step 1: Normalize to 0–255 uint8
    norm = cv2.normalize(
        anomaly_map.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX
    ).astype(np.uint8)

    # Step 2: Threshold — only keep top percentile pixels
    thresh_val = float(np.percentile(norm, threshold_percentile))
    thresh_val = max(thresh_val, 1.0)  # avoid zero threshold on flat maps
    _, binary = cv2.threshold(norm, thresh_val, 255, cv2.THRESH_BINARY)

    # Step 3: Morphological close — merge nearby blobs into one region
    if morph_kernel_size > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (morph_kernel_size, morph_kernel_size)
        )
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # Step 4: Connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )

    bboxes: List[BoundingBox] = []

    for i in range(1, num_labels):   # skip label 0 = background
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area_px:
            continue   # discard noise

        # Raw component coordinates
        cx = int(stats[i, cv2.CC_STAT_LEFT])
        cy = int(stats[i, cv2.CC_STAT_TOP])
        cw = int(stats[i, cv2.CC_STAT_WIDTH])
        ch = int(stats[i, cv2.CC_STAT_HEIGHT])

        # Expand by margin, clamp to image bounds
        x = max(0, cx - margin_px)
        y = max(0, cy - margin_px)
        x2 = min(w_map, cx + cw + margin_px)
        y2 = min(h_map, cy + ch + margin_px)
        w = x2 - x
        h = y2 - y

        # Mean anomaly score over the blob pixels (in normalised 0-1 scale)
        component_pixels = anomaly_map[labels == i]
        mean_score = float(np.mean(component_pixels))
        # Normalise score to 0–1 range relative to overall map max
        map_max = float(anomaly_map.max())
        if map_max > 0:
            mean_score = mean_score / map_max

        bboxes.append(BoundingBox(
            x=x, y=y, w=w, h=h,
            score=mean_score,
            area=area,
        ))

    # Sort by score descending — highest anomaly bbox first
    bboxes.sort(key=lambda b: b.score, reverse=True)
    return bboxes


# ── Coordinate mapping ────────────────────────────────────────────────────────

def map_bboxes_to_full_image(
    bboxes: List[BoundingBox],
    roi_offset: Tuple[int, int],
    roi_original_size: Tuple[int, int],
    model_input_size: Tuple[int, int],
) -> List[BoundingBox]:
    """
    Map bounding boxes from model input space → full aligned image space.

    When EfficientAD resizes the ROI crop to e.g. 256×256 for inference,
    the bounding box coordinates are in that 256×256 space. This function
    scales them back to the original ROI pixel size, then adds the ROI's
    offset within the full aligned image.

    Args:
        bboxes:             List of BoundingBox in model input coordinates.
        roi_offset:         (x_offset, y_offset) — top-left corner of the ROI
                            in the full aligned image.
        roi_original_size:  (w, h) of the ROI before resize to model input.
        model_input_size:   (w, h) the model was fed (e.g. (256, 256)).

    Returns:
        New list of BoundingBox with coordinates in full aligned image space.
    """
    if not bboxes:
        return []

    scale_x = roi_original_size[0] / model_input_size[0]
    scale_y = roi_original_size[1] / model_input_size[1]
    ox, oy  = roi_offset

    mapped = []
    for b in bboxes:
        mapped.append(BoundingBox(
            x     = int(b.x * scale_x) + ox,
            y     = int(b.y * scale_y) + oy,
            w     = int(b.w * scale_x),
            h     = int(b.h * scale_y),
            score = b.score,
            area  = b.area,
            label = b.label,
        ))
    return mapped


# ── Overlay helpers ───────────────────────────────────────────────────────────

def draw_bboxes(
    image: np.ndarray,
    bboxes: List[BoundingBox],
    color: Tuple[int, int, int] = (0, 0, 255),
    thickness: int = 2,
    show_score: bool = True,
    font_scale: float = 0.45,
) -> np.ndarray:
    """
    Draw bounding boxes on an image (in-place).

    Args:
        image:      BGR or grayscale image (will convert gray→BGR if needed).
        bboxes:     List of BoundingBox to draw.
        color:      BGR color tuple.
        thickness:  Rectangle border thickness.
        show_score: If True, print the score above each box.
        font_scale: Font size for score labels.

    Returns:
        Copy of the image with boxes drawn (original is not modified).
    """
    if image.ndim == 2:
        canvas = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        canvas = image.copy()

    for bbox in bboxes:
        pt1 = (bbox.x, bbox.y)
        pt2 = (bbox.x2, bbox.y2)
        cv2.rectangle(canvas, pt1, pt2, color, thickness)

        if show_score:
            label_text = f"{bbox.score:.2f}"
            text_y = max(bbox.y - 4, 12)
            cv2.putText(
                canvas, label_text,
                (bbox.x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, color, 1, cv2.LINE_AA,
            )

    return canvas


def overlay_heatmap(
    image: np.ndarray,
    anomaly_map: np.ndarray,
    alpha: float = 0.4,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """
    Blend a normalised anomaly map onto an image as a colour heatmap.

    Args:
        image:       BGR or grayscale image.
        anomaly_map: float32 H×W anomaly scores (any scale).
        alpha:       Heatmap opacity (0 = image only, 1 = heatmap only).
        colormap:    OpenCV colourmap constant (default: COLORMAP_JET).

    Returns:
        BGR image with heatmap blended in.
    """
    if image.ndim == 2:
        base = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        base = image.copy()

    h, w = base.shape[:2]

    norm = cv2.normalize(anomaly_map.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if norm.shape[:2] != (h, w):
        norm = cv2.resize(norm, (w, h), interpolation=cv2.INTER_LINEAR)

    heat = cv2.applyColorMap(norm, colormap)
    return cv2.addWeighted(base, 1.0 - alpha, heat, alpha, 0)
