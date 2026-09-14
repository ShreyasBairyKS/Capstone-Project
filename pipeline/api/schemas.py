"""
Pydantic Schemas for Part 6: Backend Service & API Layer.
Defines request and response structures for REST endpoints and WebSocket messages.
"""
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field


class ViewInspectSummary(BaseModel):
    """Inspection summary for a single camera view in the API response."""
    camera_id: str
    verdict: str
    primary_class: Optional[str] = None
    primary_confidence: Optional[float] = None
    defect_detected: bool = False
    defect_name: Optional[str] = None
    defect_confidence: Optional[float] = None
    decision_reason: str
    detection_count: int
    annotated_image_base64: Optional[str] = None


class InspectResponse(BaseModel):
    """Response payload for POST /api/inspect/image."""
    bottle_id: str
    timestamp: str
    global_verdict: str
    is_compliant: bool
    trigger_reject_actuator: bool
    primary_defect: Optional[str] = None
    primary_defect_confidence: Optional[float] = None
    flagged_cameras: List[str] = Field(default_factory=list)
    decision_reason: str
    total_latency_ms: float
    camera_count: int
    bundle_dir: Optional[str] = None
    views: Dict[str, ViewInspectSummary] = Field(default_factory=dict)


class ConfigUpdateRequest(BaseModel):
    """Payload for POST /api/config to adjust thresholds dynamically without restart."""
    pass_conf_threshold: Optional[float] = Field(None, ge=0.1, le=1.0, description="Minimum confidence for PASS")
    defect_conf_threshold: Optional[float] = Field(None, ge=0.1, le=1.0, description="Minimum confidence to trigger REJECT")
    uncertain_floor: Optional[float] = Field(None, ge=0.1, le=1.0, description="Lower floor for UNCERTAIN zone")
    save_pass_images: Optional[bool] = Field(None, description="Whether to store raw images for compliant bottles")
    mask_alpha: Optional[float] = Field(None, ge=0.0, le=1.0, description="Transparency alpha factor for masks")


class ConfigResponse(BaseModel):
    """Current live runtime configuration."""
    pass_conf_threshold: float
    defect_conf_threshold: float
    uncertain_floor: float
    save_pass_images: bool
    save_defect_images: bool
    save_yolo_labels: bool
    mask_alpha: float
    hardware_runtime: str


class DefectItem(BaseModel):
    """Summary item for GET /api/defects paginated list."""
    id: Optional[int] = None
    bottle_id: str
    timestamp: str
    epoch_time: float
    global_verdict: str
    is_compliant: bool
    trigger_actuator: bool
    primary_defect: Optional[str] = None
    primary_defect_confidence: Optional[float] = None
    flagged_cameras: List[str] = Field(default_factory=list)
    total_latency_ms: float
    bundle_dir: Optional[str] = None


class WSControlMessage(BaseModel):
    """Client-to-server command over WebSocket."""
    command: str = Field(..., description="Action: 'simulate_cycle', 'pause', 'resume', 'ping'")
    payload: Optional[Dict[str, Any]] = None
