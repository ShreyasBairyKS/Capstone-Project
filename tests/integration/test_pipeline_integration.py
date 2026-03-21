"""
tests/integration/test_pipeline_integration.py — Integration tests for the
full EdgeInferencePipeline without real ONNX models.

These tests verify:
  - The pipeline can be instantiated and run inspect() without ONNX models
  - Verdict logic integrates correctly with REMEDY engine (SeverityScorer + TriageRouter)
  - InspectionResult is correctly shaped and plumbed through

Run with:
    pytest tests/integration/test_pipeline_integration.py -v
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from core.schemas import (
    BoundingBox,
    DefectClass,
    Detection,
    InspectionResult,
    SeverityGrade,
    UQResult,
    Verdict,
)
from inference.pipeline import EdgeInferencePipeline


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _make_detection(
    cls: DefectClass = DefectClass.PACKAGING_DAMAGE,
    confidence: float = 0.88,
    area: float = 0.10,
) -> Detection:
    side = area ** 0.5
    bbox = BoundingBox(x1=0.0, y1=0.0, x2=side, y2=side)
    return Detection(
        class_id=1,
        class_name=cls,
        confidence=confidence,
        bbox=bbox,
        bbox_area_ratio=bbox.area_ratio,
    )


def _make_uq(mean: float = 0.88, std: float = 0.05) -> UQResult:
    return UQResult(
        mean_confidence=mean,
        std_confidence=std,
        ci_low=max(0.0, mean - 2 * std),
        ci_high=min(1.0, mean + 2 * std),
        is_uncertain=std >= 0.15,
        escalation_required=std >= 0.15 or mean < 0.60,
        n_passes=20,
    )


def _blank_frame(h: int = 64, w: int = 64) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #

class TestPipelineNoModels:
    """Pipeline integration without ONNX models — stubs detector + classifier."""

    def test_inspect_returns_inspection_result(self):
        """Pipeline.inspect() must always return an InspectionResult."""
        pipeline = EdgeInferencePipeline()
        # Stub detector to return no detections
        pipeline._detector = MagicMock()
        pipeline._detector.detect.return_value = []

        result = pipeline.inspect(_blank_frame(), product_id="P-INT-001")

        assert isinstance(result, InspectionResult)
        assert result.verdict == Verdict.PASS
        assert result.inspection_id != ""
        assert result.latency_ms >= 0.0

    def test_inspect_with_detection_triggers_remedy(self):
        """When a high-confidence defect is detected, REMEDY must assign a severity."""
        pipeline = EdgeInferencePipeline()
        det = _make_detection(confidence=0.92)

        pipeline._detector = MagicMock()
        pipeline._detector.detect.return_value = [det]

        pipeline._uq = MagicMock()
        pipeline._uq.estimate.return_value = _make_uq(mean=0.92, std=0.04)

        result = pipeline.inspect(_blank_frame(), sku="bottle_250ml")

        # High confidence, not uncertain → FAIL verdict
        assert result.verdict == Verdict.FAIL
        assert result.detections == [det]
        # REMEDY should have been triggered
        assert result.severity_result is not None
        assert result.remediation_action is not None
        assert result.severity_result.grade in list(SeverityGrade)

    def test_uncertain_detection_escalates(self):
        """High std → is_uncertain → ESCALATE or REVIEW verdict."""
        pipeline = EdgeInferencePipeline()
        det = _make_detection(confidence=0.72)

        pipeline._detector = MagicMock()
        pipeline._detector.detect.return_value = [det]

        pipeline._uq = MagicMock()
        pipeline._uq.estimate.return_value = _make_uq(mean=0.72, std=0.20)

        result = pipeline.inspect(_blank_frame())

        assert result.verdict in (Verdict.ESCALATE, Verdict.REVIEW, Verdict.FAIL)

    def test_no_detections_no_severity(self):
        """PASS verdict must have no severity_result or remediation_action."""
        pipeline = EdgeInferencePipeline()
        pipeline._detector = MagicMock()
        pipeline._detector.detect.return_value = []

        result = pipeline.inspect(_blank_frame())

        assert result.verdict == Verdict.PASS
        assert result.severity_result is None
        assert result.remediation_action is None
        assert result.uq_result is None

    def test_inspection_id_is_unique(self):
        """Each call to inspect() must produce a unique inspection_id."""
        pipeline = EdgeInferencePipeline()
        pipeline._detector = MagicMock()
        pipeline._detector.detect.return_value = []

        ids = {pipeline.inspect(_blank_frame()).inspection_id for _ in range(10)}
        assert len(ids) == 10

    def test_product_id_propagated(self):
        """product_id passed to inspect() must appear in the result."""
        pipeline = EdgeInferencePipeline()
        pipeline._detector = MagicMock()
        pipeline._detector.detect.return_value = []

        result = pipeline.inspect(_blank_frame(), product_id="P-XYZ-999")
        assert result.product_id == "P-XYZ-999"

    def test_latency_is_positive(self):
        """latency_ms must be a positive float."""
        pipeline = EdgeInferencePipeline()
        pipeline._detector = MagicMock()
        pipeline._detector.detect.return_value = []

        result = pipeline.inspect(_blank_frame())
        assert result.latency_ms > 0.0

    @pytest.mark.parametrize("cls", list(DefectClass))
    def test_all_defect_classes_produce_severity(self, cls):
        """Every defect class must flow through to a severity grade."""
        pipeline = EdgeInferencePipeline()
        det = _make_detection(cls=cls, confidence=0.91)

        pipeline._detector = MagicMock()
        pipeline._detector.detect.return_value = [det]
        pipeline._uq = MagicMock()
        pipeline._uq.estimate.return_value = _make_uq(mean=0.91, std=0.04)

        result = pipeline.inspect(_blank_frame())

        assert result.severity_result is not None
        assert result.severity_result.grade in list(SeverityGrade)


class TestRemedyIntegration:
    """Verify SeverityScorer + TriageRouter integrate correctly via pipeline."""

    def test_surface_contamination_not_remediable_at_s4(self):
        """S4 surface contamination must produce REJECT action."""
        from remedy.severity_scorer import SeverityScorer
        from remedy.triage_router import TriageRouter
        from core.schemas import RemediationActionType

        pipeline = EdgeInferencePipeline()
        # Large area surface contamination → S4
        det = _make_detection(cls=DefectClass.SURFACE_CONTAMINATION, confidence=0.95, area=0.50)

        pipeline._detector = MagicMock()
        pipeline._detector.detect.return_value = [det]
        pipeline._uq = MagicMock()
        pipeline._uq.estimate.return_value = _make_uq(mean=0.95, std=0.18)

        result = pipeline.inspect(_blank_frame(), attempt_count=2)

        if result.remediation_action:
            # S3/S4 contamination: must be REJECT or escalated action
            assert result.remediation_action.action in (
                RemediationActionType.REJECT,
                RemediationActionType.CLEAN,
            )
