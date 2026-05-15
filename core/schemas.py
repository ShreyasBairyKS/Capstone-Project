"""
core/schemas.py — Shared Pydantic models used across all VisionFood QAI modules.

These are the canonical data contracts. API response schemas, ORM adapters,
and all internal functions should map to/from these types.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------- #
# Enumerations
# --------------------------------------------------------------------------- #


class DefectClass(str, Enum):
    IMPROPER_FILLING = "improper_filling"
    PACKAGING_DAMAGE = "packaging_damage"
    LABEL_MISALIGNMENT = "label_misalignment"
    SURFACE_CONTAMINATION = "surface_contamination"
    FILL_LEVEL_LOW = "fill_level_low"
    FILL_LEVEL_HIGH = "fill_level_high"
    CAP_FITTING_ANOMALY = "cap_fitting_anomaly"
    SURFACE_TEAR = "surface_tear"
    SURFACE_SMUDGE = "surface_smudge"
    LABEL_DATE_MISMATCH = "label_date_mismatch"
    LABEL_BARCODE_MISMATCH = "label_barcode_mismatch"


# Canonical ordered list — single source of truth for class index ↔ name mapping
DEFECT_CLASS_NAMES: list[str] = [c.value for c in DefectClass]


class Verdict(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    ESCALATE = "ESCALATE"
    REVIEW = "REVIEW"


class ProductCategory(str, Enum):
    BEVERAGE = "beverage"
    FOOD = "food"
    GENERAL = "general"


class ProductSubType(str, Enum):
    TRANSPARENT_BOTTLE = "transparent_bottle"
    RIGID_CAN = "rigid_can"
    FLEXIBLE_WRAPPER = "flexible_wrapper"
    RIGID_BOX = "rigid_box"


class ContainerContents(str, Enum):
    LIQUID = "liquid"
    SOLID = "solid"


class SeverityGrade(str, Enum):
    S1 = "S1"  # Minor — remediable on line
    S2 = "S2"  # Moderate — remediable with station intervention
    S3 = "S3"  # Severe — reject
    S4 = "S4"  # Critical — immediate quarantine


class RemediationActionType(str, Enum):
    RELABEL = "RELABEL"
    REFILL = "REFILL"
    REPACK = "REPACK"
    CLEAN = "CLEAN"
    REJECT = "REJECT"
    PASS = "PASS"


# --------------------------------------------------------------------------- #
# Detection schemas
# --------------------------------------------------------------------------- #


class BoundingBox(BaseModel):
    """Normalised bounding box coordinates [0.0, 1.0]."""

    x1: float = Field(..., ge=0.0, le=1.0)
    y1: float = Field(..., ge=0.0, le=1.0)
    x2: float = Field(..., ge=0.0, le=1.0)
    y2: float = Field(..., ge=0.0, le=1.0)

    @property
    def area_ratio(self) -> float:
        return max(0.0, (self.x2 - self.x1) * (self.y2 - self.y1))

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)


class Detection(BaseModel):
    """Single defect detection from YOLOv11."""

    class_id: int
    class_name: DefectClass
    confidence: float = Field(..., ge=0.0, le=1.0)
    bbox: BoundingBox
    bbox_area_ratio: float = Field(default=0.0, ge=0.0, le=1.0)

    @classmethod
    def from_yolo_output(
        cls,
        class_id: int,
        confidence: float,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
    ) -> "Detection":
        bbox = BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)
        class_names = list(DefectClass)
        return cls(
            class_id=class_id,
            class_name=class_names[class_id],
            confidence=confidence,
            bbox=bbox,
            bbox_area_ratio=bbox.area_ratio,
        )


# --------------------------------------------------------------------------- #
# Uncertainty Quantification schema
# --------------------------------------------------------------------------- #


class UQResult(BaseModel):
    """Output from MC Dropout uncertainty quantification."""

    mean_confidence: float = Field(..., ge=0.0, le=1.0)
    std_confidence: float = Field(..., ge=0.0)
    ci_low: float = Field(..., ge=0.0, le=1.0)
    ci_high: float = Field(..., ge=0.0, le=1.0)
    is_uncertain: bool
    escalation_required: bool
    n_passes: int = 20


# --------------------------------------------------------------------------- #
# REMEDY engine schemas
# --------------------------------------------------------------------------- #


class SeverityResult(BaseModel):
    """Output from the SeverityScorer."""

    grade: SeverityGrade
    score: float = Field(..., ge=0.0, le=1.0)
    # Score component breakdown (sum to 1.0 weighted)
    area_component: float
    conf_uncertainty_component: float
    class_risk_component: float
    attempt_penalty_component: float


class RemediationAction(BaseModel):
    """Triage decision from the TriageRouter."""

    action: RemediationActionType
    station: Optional[str] = None  # "A", "B", "C" or None if reject/pass
    is_remediable: bool
    reason: str
    max_attempts: int = 2


# --------------------------------------------------------------------------- #
# Top-level inspection result
# --------------------------------------------------------------------------- #


class InspectionResult(BaseModel):
    """
    Complete result from a single product inspection.
    Produced by EdgeInferencePipeline.inspect() and persisted to the database.
    """

    inspection_id: str
    product_id: Optional[str] = None
    sku: str = "default"
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    product_category: Optional[ProductCategory] = None
    product_sub_type: Optional[ProductSubType] = None
    container_contents: Optional[ContainerContents] = None

    # Verdict
    verdict: Verdict
    escalated: bool = False

    # Detections (empty list means clean product)
    detections: list[Detection] = []

    # Uncertainty quantification (populated when detections exist)
    uq_result: Optional[UQResult] = None

    # REMEDY outputs (populated when verdict is FAIL or ESCALATE)
    severity_result: Optional[SeverityResult] = None
    remediation_action: Optional[RemediationAction] = None

    # Optional presentation/diagnostic payloads for dashboard inspection views.
    annotated_image_b64: Optional[str] = None
    inference_summary: Optional[dict[str, Any]] = None

    # Performance metadata
    latency_ms: float = 0.0
    device_id: str = "edge_node_01"

    @property
    def has_defects(self) -> bool:
        return len(self.detections) > 0

    @property
    def primary_defect(self) -> Optional[Detection]:
        """Highest-confidence detection, or None if clean."""
        if not self.detections:
            return None
        return max(self.detections, key=lambda d: d.confidence)
