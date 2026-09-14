"""
Data schemas for Part 2: Multi-View Inspection Decision Logic (The Quality Gate).
"""
from enum import Enum
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field

from pipeline.inference.schemas import Detection, CameraResult


class InspectionVerdict(str, Enum):
    PASS = "PASS"                        # Certified compliant bottle cap across all angles
    REJECT = "REJECT"                    # Defect identified on at least one camera view
    UNCERTAIN = "UNCERTAIN"              # Borderline confidence; routed for review without false rejection
    INSPECTION_FAILED = "INSPECTION_FAILED" # No cap detected or severe anomaly


class ViewVerdict(BaseModel):
    """Inspection outcome for an individual camera angle."""
    camera_id: str = Field(..., description="Camera view identifier")
    verdict: InspectionVerdict = Field(..., description="Verdict for this specific view")
    primary_class: Optional[str] = Field(None, description="Top class detected in this view")
    primary_confidence: Optional[float] = Field(None, description="Confidence of top class")
    defect_detected: bool = Field(default=False, description="True if a defect was identified in this view")
    defect_name: Optional[str] = Field(None, description="Name of defect if present")
    defect_confidence: Optional[float] = Field(None, description="Confidence of defect detection")
    decision_reason: str = Field(..., description="Explanatory reason for this view verdict")
    detections: List[Detection] = Field(default_factory=list, description="All detections in this view")


class GlobalInspectionResult(BaseModel):
    """
    Aggregated multi-view inspection result for a single bottle instance.
    This is the authoritative decision consumed by the physical reject actuator,
    the audit logger, and the operator dashboard.
    """
    bottle_id: str = Field(..., description="Unique inspection session ID for this bottle")
    timestamp: str = Field(..., description="ISO formatted timestamp of inspection")
    global_verdict: InspectionVerdict = Field(..., description="Final aggregated manufacturing verdict")
    is_compliant: bool = Field(..., description="True if PASS, False otherwise")
    trigger_reject_actuator: bool = Field(..., description="Hardware flag: True if bottle must be ejected")
    primary_defect: Optional[str] = Field(None, description="Defect type that triggered rejection")
    primary_defect_confidence: Optional[float] = Field(None, description="Confidence of primary defect")
    flagged_cameras: List[str] = Field(default_factory=list, description="Cameras that observed the defect/issue")
    decision_reason: str = Field(..., description="High-level explanation of global verdict")
    total_latency_ms: float = Field(..., description="Inference + decision evaluation latency in ms")
    per_view_results: Dict[str, ViewVerdict] = Field(default_factory=dict, description="View verdicts by camera_id")

    @property
    def camera_count(self) -> int:
        return len(self.per_view_results)
