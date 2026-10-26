"""
Data schemas for Part 2: Fill Quality Gate and Decision Logic.
Defines authoritative verdicts and structured inspection outcome models.
"""
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field

from pipeline.fill_level.inference.schemas import BottleBox, MeniscusBox


class FillVerdict(str, Enum):
    PASS = "PASS"                           # Liquid volume within nominal tolerance (40% - 85%)
    REJECT = "REJECT"                       # Underfill, overfill, or dry/empty container
    UNCERTAIN = "UNCERTAIN"                 # Borderline liquid level requiring operator review
    INSPECTION_FAILED = "INSPECTION_FAILED" # No bottle detected (empty conveyor gap)


class FillInspectionDecision(BaseModel):
    """
    Authoritative manufacturing inspection decision for a single bottle profile image.
    Consumed by the physical reject actuator, SQLite audit logger, and the operator dashboard.
    """
    bottle_id: str = Field(..., description="Unique inspection identifier for this bottle")
    timestamp: str = Field(..., description="ISO 8601 formatted timestamp")
    verdict: FillVerdict = Field(..., description="Authoritative quality verdict")
    is_compliant: bool = Field(..., description="True if certified PASS, False otherwise")
    trigger_reject_actuator: bool = Field(..., description="Hardware flag: True if bottle must be ejected")
    fill_status: str = Field(..., description="Detailed status: 'normal', 'underfill', 'overfill', 'no_bottle', 'no_water'")
    fill_ratio: Optional[float] = Field(None, description="Calculated normalized liquid fill ratio 0.0 to 1.0")
    decision_reason: str = Field(..., description="Human-readable explanation of verdict")
    decision_latency_ms: float = Field(..., description="Time spent evaluating decision rules (ms)")
    total_latency_ms: float = Field(..., description="Total latency (inference + decision evaluation) in ms")
    bottle: Optional[BottleBox] = Field(None, description="Bottle bounding box")
    meniscus: Optional[MeniscusBox] = Field(None, description="Liquid meniscus bounding box")
