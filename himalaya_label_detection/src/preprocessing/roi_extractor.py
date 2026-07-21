"""
roi_extractor.py
Extracts functional ROI crops from an aligned label image.

ROI coordinates are loaded from config/roi_config.yaml which is
populated interactively by scripts/setup_golden_master.py.

Why functional ROI naming (not positional Top/Center/Bottom):
- Layout-change resilience: if Himalaya moves the barcode, only the
  YAML config needs updating — the pipeline and model names stay the same.
- Operator-readable failure reports: "ROI_BARCODE anomaly" is actionable;
  "Bottom ROI anomaly" is not.
"""

import cv2
import numpy as np
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple


@dataclass
class ROIConfig:
    name: str
    x: int
    y: int
    w: int
    h: int
    critical: bool = False
    description: str = ""


@dataclass
class ROIExtractionResult:
    crops: Dict[str, np.ndarray]          # roi_name → crop array
    valid: Dict[str, bool]                # roi_name → True if crop is non-empty
    missing: list[str]                    # ROIs that could not be extracted


# Canonical ROI names — used as keys throughout the entire pipeline
ALL_ROI_NAMES = [
    "ROI_LOGO",
    "ROI_PRODUCT_NAME",
    "ROI_INGREDIENT_BAR",
    "ROI_GRAPHICS",
    "ROI_CLAIMS",
    "ROI_BARCODE",
    "ROI_BATCH_CODE",
    "ROI_CERT",
    "ROI_TEXT_BLOCK",
]


def load_roi_configs(config_path: str | Path) -> Dict[str, ROIConfig]:
    """
    Load ROI definitions from YAML.

    Returns:
        Dict mapping ROI name → ROIConfig.

    Raises:
        ValueError: If the config has not been set up yet (configured: false).
    """
    config_path = Path(config_path)
    with open(config_path) as f:
        data = yaml.safe_load(f)

    if not data.get("configured", False):
        raise ValueError(
            "ROI config has not been set up yet.\n"
            "Run:  python scripts/setup_golden_master.py\n"
            "to define ROI bounding boxes interactively."
        )

    configs: Dict[str, ROIConfig] = {}
    for name, attrs in data["rois"].items():
        x, y, w, h = attrs["coords"]
        configs[name] = ROIConfig(
            name=name,
            x=x, y=y, w=w, h=h,
            critical=attrs.get("critical", False),
            description=attrs.get("description", ""),
        )
    return configs


class ROIExtractor:
    """
    Crops functional regions from an aligned label image.

    Usage:
        extractor = ROIExtractor("config/roi_config.yaml")
        result    = extractor.extract(aligned_image)
        logo_crop = result.crops["ROI_LOGO"]
    """

    def __init__(self, config_path: str | Path,
                 resize_each_roi: Optional[Tuple[int, int]] = None):
        """
        Args:
            config_path:     Path to roi_config.yaml.
            resize_each_roi: Optional (width, height) to resize every ROI crop to.
                             Useful to normalise input sizes for the anomaly models.
        """
        self.configs = load_roi_configs(config_path)
        self.resize_each_roi = resize_each_roi

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self, aligned_image: np.ndarray) -> ROIExtractionResult:
        """
        Extract all ROI crops from an aligned label image.

        Args:
            aligned_image: numpy array warped to the golden master coordinate system.

        Returns:
            ROIExtractionResult with crop arrays and validity flags.
        """
        img_h, img_w = aligned_image.shape[:2]
        crops: Dict[str, np.ndarray] = {}
        valid: Dict[str, bool] = {}
        missing: list[str] = []

        for name, cfg in self.configs.items():
            # Clamp coordinates to image boundaries
            x1 = max(0, cfg.x)
            y1 = max(0, cfg.y)
            x2 = min(img_w, cfg.x + cfg.w)
            y2 = min(img_h, cfg.y + cfg.h)

            if x2 <= x1 or y2 <= y1:
                # ROI is outside image bounds entirely
                missing.append(name)
                valid[name] = False
                crops[name] = np.zeros((cfg.h, cfg.w, 3), dtype=np.uint8)
                continue

            crop = aligned_image[y1:y2, x1:x2].copy()

            if self.resize_each_roi is not None:
                crop = cv2.resize(crop, self.resize_each_roi, interpolation=cv2.INTER_AREA)

            crops[name] = crop
            valid[name] = True

        return ROIExtractionResult(crops=crops, valid=valid, missing=missing)

    def draw_rois(self, image: np.ndarray,
                  highlight: Optional[list[str]] = None) -> np.ndarray:
        """
        Draw all ROI bounding boxes on a copy of the image for inspection.

        Args:
            image:     The aligned label image.
            highlight: Optional list of ROI names to draw in red (others are green).

        Returns:
            Annotated BGR image.
        """
        canvas = image.copy() if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        highlight = set(highlight or [])

        for name, cfg in self.configs.items():
            color = (0, 0, 220) if name in highlight else (0, 200, 0)
            thickness = 2 if name not in highlight else 3

            cv2.rectangle(canvas,
                          (cfg.x, cfg.y),
                          (cfg.x + cfg.w, cfg.y + cfg.h),
                          color, thickness)

            # Label text above the box
            text_y = max(cfg.y - 6, 14)
            cv2.putText(canvas, name.replace("ROI_", ""),
                        (cfg.x + 3, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1,
                        cv2.LINE_AA)

        return canvas

    @property
    def critical_rois(self) -> list[str]:
        """Return names of ROIs marked as critical."""
        return [name for name, cfg in self.configs.items() if cfg.critical]
