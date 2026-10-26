"""
Visualizer configuration parameters for Fill Level Station.
"""
from pydantic import BaseModel, Field

from pipeline.fill_level.config import fill_threshold_from_env


class FillVisualizerConfig(BaseModel):
    """
    Configuration options for on-demand dynamic visual overlay rendering.
    Draws non-destructive annotations on memory buffers without modifying stored raw images.
    """
    draw_bottle_box: bool = Field(default=True, description="Draw bounding box around detected bottle")
    draw_meniscus_line: bool = Field(default=True, description="Draw glowing meniscus line at air-liquid interface")
    draw_ruler: bool = Field(default=True, description="Draw calibrated vertical fill measurement ruler")
    draw_status_banner: bool = Field(default=True, description="Draw high-contrast status badge on frame")
    jpeg_quality: int = Field(default=85, ge=50, le=100, description="JPEG quality for live stream transmission")
    ruler_width_px: int = Field(default=32, description="Width in pixels of the vertical measurement ruler")
    underfill_thresh: float = Field(default_factory=lambda: fill_threshold_from_env("VISIONQAI_FILL_UNDERFILL_THRESHOLD", 0.40), description="Underfill ratio boundary for ruler marking")
    overfill_thresh: float = Field(default_factory=lambda: fill_threshold_from_env("VISIONQAI_FILL_OVERFILL_THRESHOLD", 0.85), description="Overfill ratio boundary for ruler marking")
