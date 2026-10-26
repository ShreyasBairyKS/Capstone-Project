"""
API request and response schemas for Station 2 (Fill Level Inspection).
"""
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class FillInspectResponse(BaseModel):
    """Manufacturing telemetry response for a single bottle fill inspection."""
    bottle_id: str = Field(..., description="Unique bottle inspection identifier")
    timestamp: str = Field(..., description="ISO formatted timestamp")
    verdict: str = Field(..., description="'PASS', 'REJECT', 'UNCERTAIN', 'INSPECTION_FAILED'")
    is_compliant: bool = Field(..., description="True if certified PASS, False otherwise")
    trigger_reject_actuator: bool = Field(..., description="Pneumatic reject actuator status")
    fill_status: str = Field(..., description="'normal', 'underfill', 'overfill', 'no_bottle', 'no_water'")
    fill_ratio: Optional[float] = Field(None, description="Normalized liquid volume ratio 0.0 to 1.0")
    decision_reason: str = Field(..., description="Human-readable decision explanation")
    total_latency_ms: float = Field(..., description="End-to-end forward pass + gate latency in ms")
    inference_latency_ms: float = Field(default=0.0, description="Forward pass time in ms")
    annotated_image_base64: Optional[str] = Field(None, description="Data URL of rendered overlay")
    bottle_box: Optional[List[int]] = Field(None, description="[x1, y1, x2, y2] of container")
    bottle_confidence: Optional[float] = Field(None, description="Confidence of bottle localizer")
    meniscus_box: Optional[List[int]] = Field(None, description="[wx1, wy1, wx2, wy2] of liquid surface")
    meniscus_confidence: Optional[float] = Field(None, description="Confidence of meniscus detector")
    water_y: Optional[int] = Field(None, description="Vertical pixel midpoint of liquid surface")
    bundle_dir: Optional[str] = Field(None, description="Path to persisted disk bundle")


class FillActiveLearningCandidate(BaseModel):
    """Candidate inspection item displayed in the S3 Active Learning triage queue."""
    bottle_id: str
    timestamp: str
    epoch_time: float
    global_verdict: str
    fill_status: str
    fill_ratio: Optional[float] = None
    meniscus_confidence: Optional[float] = None
    has_labels: bool = False
    is_uncertain: bool = False


class SaveFillAnnotationRequest(BaseModel):
    """Payload submitted when operator commits ground-truth adjustment in S3 Active Learning."""
    bottle_id: str
    global_verdict: Optional[str] = None
    fill_status: Optional[str] = None
    water_box: Optional[List[int]] = None  # [x1, y1, x2, y2] adjusted in canvas


class FillConfigUpdateRequest(BaseModel):
    """Partial update of Station 2 runtime thresholds (omitted fields are unchanged)."""
    underfill_threshold: Optional[float] = Field(None, description="Min fill ratio (0.10-0.60)")
    overfill_threshold: Optional[float] = Field(None, description="Max fill ratio (0.60-0.98)")
    borderline_band: Optional[float] = Field(None, description="Gray-zone half-width around thresholds (0.00-0.05)")
    borderline_conf_floor: Optional[float] = Field(None, description="Meniscus confidence below which borderline -> UNCERTAIN (0.20-0.80)")
    bottle_conf_threshold: Optional[float] = Field(None, ge=0.05, le=0.95, description="Bottle detector confidence")
    water_conf_threshold: Optional[float] = Field(None, ge=0.05, le=0.95, description="Meniscus detector confidence")


class FillConfigResponse(BaseModel):
    """Active Station 2 runtime configuration."""
    underfill_threshold: float
    overfill_threshold: float
    borderline_band: float
    borderline_conf_floor: float
    bottle_conf_threshold: float
    water_conf_threshold: float
    hardware_runtime: str


class FillHealthResponse(BaseModel):
    """Status report for Station 2 inference engine and GPU VRAM allocation."""
    status: str = "ok"
    station: str = "Station 2: Fill Level Inspection"
    models_loaded: bool = Field(..., description="True if models are allocated in VRAM, False if in standby")
    hardware_runtime: str = Field(..., description="Active device backend (NVIDIA GPU / CPU)")
    engine_version: str = Field(..., description="Active TensorRT engine version string")
    underfill_threshold: float = Field(..., description="Minimum fill ratio tolerance")
    overfill_threshold: float = Field(..., description="Maximum fill ratio tolerance")
