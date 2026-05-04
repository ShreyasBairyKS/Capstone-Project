"""
inference/bottle_cap_v3/config.py
==================================
Configuration for V3 Bottle-Guided Cap Detection pipeline.

Tune these values for your specific camera setup and deployment target.
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class V3Config:
    """V3 pipeline configuration."""

    # ── Model ──
    weights: str = "runs/detect/bottle_cap_det_v2/weights/best.pt"
    device: str = "0"            # "0" = GPU, "cpu" = CPU

    # ── Detection thresholds ──
    det_conf: float = 0.25       # Confidence threshold for pass 1 (full frame)
    cap_conf: float = 0.15       # LOWER threshold for pass 2 (zoomed cap crops)
    #                              ↑ Lower because caps are harder to detect
    iou_thresh: float = 0.45     # NMS IoU threshold

    # ── Bottle-guided crop settings ──
    # What fraction of the bottle bounding box (from the TOP) to crop for cap search.
    # Caps are always at the top of bottles.
    #   0.35 = crop top 35% of bottle bbox  (good default)
    #   0.50 = crop top 50%  (if caps are large or bottles are short)
    cap_region_ratio: float = 0.35

    # Context padding around the crop (fraction of crop size)
    # Adds extra pixels around the crop to give the model more context
    crop_pad: float = 0.15

    # Minimum crop size in pixels (skip tiny bottles)
    min_crop_px: int = 32

    # ── Scale factor for zoomed crops ──
    # Upscale the crop before feeding to model for better small-object detection
    # 2.0 = double the crop resolution (like 2x digital zoom)
    # 3.0 = triple (more accurate but slower)
    # This is the KEY parameter that solves the small-cap problem
    zoom_scale: float = 2.5

    # Maximum zoom crop size (prevents OOM on very large bottles)
    max_zoom_px: int = 960

    # ── Class name matching ──
    bottle_keyword: str = "bottle"
    cap_keyword: str = "cap"

    # ── Output ──
    output_dir: str = "runs/bottle_cap_v3"
    save_annotated: bool = True
    save_crops: bool = False     # Save individual cap crops for debugging
    save_json: bool = True

    # ── Display ──
    show_window: bool = False
    window_width: int = 1280

    # ── Performance (Jetson-specific) ──
    # Maximum number of bottles to zoom into per frame
    # On Jetson, limit to 3-4 to maintain FPS
    max_zoom_targets: int = 5

    # Skip pass 2 if pass 1 already found caps with high confidence
    skip_zoom_if_cap_conf: float = 0.70  # If any cap >70% in pass 1, skip zoom


@dataclass
class JetsonConfig(V3Config):
    """Optimized config for NVIDIA Jetson deployment."""
    det_conf: float = 0.30       # Slightly higher to reduce false positives
    cap_conf: float = 0.20
    zoom_scale: float = 2.0      # Lower zoom for speed
    max_zoom_px: int = 640
    max_zoom_targets: int = 3
    device: str = "0"


@dataclass
class DebugConfig(V3Config):
    """Config for debugging — saves everything."""
    det_conf: float = 0.15
    cap_conf: float = 0.10
    zoom_scale: float = 3.0
    save_crops: bool = True
    save_annotated: bool = True
    save_json: bool = True


# Preset configs
PRESETS = {
    "default": V3Config,
    "jetson": JetsonConfig,
    "debug": DebugConfig,
}
