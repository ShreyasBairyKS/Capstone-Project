"""
Schemas for Part 1: Core Inference Engine
Defines typed data models for raw detections, camera results, and multi-camera aggregates.
"""
from typing import List, Dict, Tuple, Optional, Any
from pydantic import BaseModel, Field


class Detection(BaseModel):
    """Represents a single detected object in a frame."""
    class_id: int = Field(..., description="Numeric class ID (0-5)")
    class_name: str = Field(..., description="Human-readable class name (e.g. good_cap, damaged_cap)")
    confidence: float = Field(..., description="Raw detection confidence (0.0 to 1.0)")
    box: List[float] = Field(..., description="Bounding box [x1, y1, x2, y2] in pixel coordinates")
    mask: Optional[List[List[float]]] = Field(
        default=None, 
        description="Polygon segmentation contour [[x, y], ...] in pixel coordinates"
    )
    mask_area_pixels: Optional[float] = Field(
        default=None,
        description="Calculated pixel area of the segmentation mask"
    )


class CameraResult(BaseModel):
    """Inference result for an individual camera view."""
    camera_id: str = Field(..., description="Identifier of the camera (e.g. camera_front, camera_rear)")
    frame_width: int = Field(..., description="Frame width in pixels")
    frame_height: int = Field(..., description="Frame height in pixels")
    detections: List[Detection] = Field(default_factory=list, description="List of detected objects in this view")
    inference_latency_ms: float = Field(..., description="Forward pass time for this batch/item (ms)")

    @property
    def detection_count(self) -> int:
        return len(self.detections)

    @property
    def has_detections(self) -> bool:
        return len(self.detections) > 0


class MultiCameraInferenceResult(BaseModel):
    """Aggregated inference output across all camera views for a single bottle inspection."""
    hardware_used: str = Field(..., description="Device and backend used (e.g. NVIDIA GPU (CUDA), CPU Fallback)")
    total_latency_ms: float = Field(..., description="Total wall-clock latency for batched inference (ms)")
    camera_count: int = Field(..., description="Number of camera views processed")
    views: Dict[str, CameraResult] = Field(
        default_factory=dict, 
        description="Mapping of camera_id -> CameraResult"
    )

    def get_view(self, camera_id: str) -> Optional[CameraResult]:
        return self.views.get(camera_id)
