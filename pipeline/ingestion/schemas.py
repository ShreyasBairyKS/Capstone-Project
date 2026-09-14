"""
Schemas for Part 3: Stream & Ingestion Manager (Multi-Camera Array).
Defines camera configurations, frame bundles, and stream metadata.
"""
from typing import Dict, Optional, List, Union, Any, Literal
import numpy as np
from pydantic import BaseModel, Field


class CameraSourceConfig(BaseModel):
    """Configuration for an individual camera in the array."""
    camera_id: str = Field(..., description="Unique camera identifier (e.g. camera_front, camera_rear)")
    source_type: Literal["device", "video", "rtsp", "image_pair"] = Field(
        default="device",
        description="Type of video/image input source"
    )
    source_path: Union[int, str] = Field(
        ...,
        description="Device index (0, 1) or file path or RTSP URL"
    )
    width: Optional[int] = Field(default=None, description="Requested capture width")
    height: Optional[int] = Field(default=None, description="Requested capture height")
    fps: Optional[int] = Field(default=30, description="Target capture FPS")
    roi: Optional[List[int]] = Field(
        default=None,
        description="Optional Region of Interest [ymin, xmin, ymax, xmax] in normalized or pixel coordinates"
    )


class SynchronizedFrameBundle:
    """
    A time-synchronized bundle of raw image frames captured simultaneously
    from all cameras for a single bottle inspection event.
    """
    def __init__(
        self,
        bottle_id: str,
        timestamp: str,
        frames: Dict[str, np.ndarray],
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.bottle_id = bottle_id
        self.timestamp = timestamp
        self.frames = frames  # Mapping of camera_id -> np.ndarray (BGR image)
        self.metadata = metadata or {}

    @property
    def camera_count(self) -> int:
        return len(self.frames)

    def get_frame(self, camera_id: str) -> Optional[np.ndarray]:
        return self.frames.get(camera_id)

    def __repr__(self) -> str:
        cam_info = ", ".join(f"{k}:{v.shape}" for k, v in self.frames.items())
        return f"<SynchronizedFrameBundle bottle_id='{self.bottle_id}' cameras=[{cam_info}]>"
