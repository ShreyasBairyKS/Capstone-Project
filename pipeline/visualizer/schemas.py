"""
Configuration and data schemas for Part 4: Visual Overlay & Annotation Engine.
Defines color palettes, rendering toggles, and layout options.
"""
from typing import Dict, Tuple, Optional
from pydantic import BaseModel, Field


# Standard BGR color tuples (Blue, Green, Red)
DEFAULT_BGR_PALETTE: Dict[str, Tuple[int, int, int]] = {
    # Classes
    "good_cap": (113, 204, 46),       # Emerald Green
    "damaged_cap": (46, 46, 231),      # Crimson Red
    "misplaced_cap": (46, 120, 231),    # Coral Orange-Red
    "open_cap": (30, 70, 240),         # Bright Red
    "wet_cap": (235, 160, 52),         # Sky/Teal Blue
    "no_cap": (30, 30, 200),           # Deep Red
    
    # Decisions / Status Banners
    "PASS": (113, 204, 46),            # Green
    "REJECT": (46, 46, 231),           # Red
    "UNCERTAIN": (0, 191, 255),        # Deep Yellow / Amber
    "INSPECTION_FAILED": (100, 100, 100) # Neutral Gray
}


class VisualizerConfig(BaseModel):
    """Configuration options for the visual annotation and HUD overlay renderer."""
    mask_alpha: float = Field(default=0.40, ge=0.0, le=1.0, description="Transparency alpha blend factor for segmentation masks")
    mask_border_thickness: int = Field(default=2, ge=0, le=10, description="Border outline thickness for polygon masks")
    box_thickness: int = Field(default=2, ge=1, le=8, description="Bounding box outline thickness")
    font_scale: float = Field(default=0.55, ge=0.2, le=2.0, description="OpenCV font scale for labels")
    font_thickness: int = Field(default=1, ge=1, le=4, description="OpenCV font thickness")
    show_hud_banner: bool = Field(default=True, description="Whether to render high-visibility top status banner")
    show_confidence: bool = Field(default=True, description="Whether to display percentage on defect pills")
    show_boxes: bool = Field(default=True, description="Whether to draw bounding box rectangles")
    show_masks: bool = Field(default=True, description="Whether to blend semi-transparent polygon masks")
    show_camera_label: bool = Field(default=True, description="Whether to show camera ID tag in bottom-left corner")
    side_by_side_max_width: int = Field(default=1920, ge=640, le=4096, description="Max width for stitched multi-camera review grid")
    color_palette: Dict[str, Tuple[int, int, int]] = Field(
        default_factory=lambda: DEFAULT_BGR_PALETTE.copy(),
        description="Mapping of class name or status string to BGR color tuple"
    )

    def get_color(self, name: str) -> Tuple[int, int, int]:
        """Returns BGR color tuple for class name or status."""
        return self.color_palette.get(name, (0, 255, 255))  # Default to Yellow
