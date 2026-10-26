"""
Data schemas for Part 1: Core Fill Level Inference Engine.
Defines typed models for single-camera bottle detection, meniscus localization, and fill ratio outcomes.
"""
from typing import List, Optional
from pydantic import BaseModel, Field


class BottleBox(BaseModel):
    """Bounding box coordinates and confidence of a detected bottle."""
    box: List[int] = Field(..., description="[x1, y1, x2, y2] in pixel coordinates")
    confidence: float = Field(..., description="Detection confidence (0.0 to 1.0)")

    @property
    def width(self) -> int:
        return max(0, self.box[2] - self.box[0])

    @property
    def height(self) -> int:
        return max(0, self.box[3] - self.box[1])


class MeniscusBox(BaseModel):
    """Bounding box coordinates and center Y of detected liquid meniscus surface line."""
    box: List[int] = Field(..., description="[wx1, wy1, wx2, wy2] in pixel coordinates")
    water_y: int = Field(..., description="Calculated center Y line of the meniscus")
    confidence: float = Field(..., description="Detection confidence (0.0 to 1.0)")


class SingleImageFillResult(BaseModel):
    """Aggregated fill level inference outcome for a single bottle profile image."""
    hardware_used: str = Field(..., description="Hardware backend (NVIDIA GPU / CPU)")
    inference_latency_ms: float = Field(..., description="Total forward pass time in ms")
    bottle: Optional[BottleBox] = Field(None, description="Bottle bounding box if located")
    meniscus: Optional[MeniscusBox] = Field(None, description="Meniscus line if located")
    fill_ratio: Optional[float] = Field(None, description="Normalized liquid fill ratio 0.0 to 1.0")
    fill_verdict: str = Field(
        ..., 
        description="Fill level verdict: 'normal', 'underfill', 'overfill', 'no_bottle', 'no_water'"
    )
    water_y: Optional[int] = Field(None, description="Center Y of meniscus in image coords")
    bottle_h: Optional[int] = Field(None, description="Height of bottle in pixels")

    @property
    def is_detected(self) -> bool:
        return self.bottle is not None and self.meniscus is not None
