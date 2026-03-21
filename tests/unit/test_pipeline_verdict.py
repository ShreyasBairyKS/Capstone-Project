"""
tests/unit/test_pipeline_verdict.py — Unit tests for EdgeInferencePipeline._apply_verdict_logic.

Tests the verdict logic in isolation (no ONNX models required).
"""

from __future__ import annotations

import pytest

from core.schemas import (
    BoundingBox,
    DefectClass,
    Detection,
    UQResult,
    Verdict,
)
from inference.pipeline import EdgeInferencePipeline


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

def _make_detection(confidence: float = 0.90) -> Detection:
    bbox = BoundingBox(x1=0.1, y1=0.1, x2=0.3, y2=0.3)
    return Detection(
        class_id=0,
        class_name=DefectClass.LABEL_MISALIGNMENT,
        confidence=confidence,
        bbox=bbox,
        bbox_area_ratio=bbox.area_ratio,
    )


def _make_uq(mean: float, std: float = 0.05) -> UQResult:
    return UQResult(
        mean_confidence=mean,
        std_confidence=std,
        ci_low=max(0.0, mean - 2 * std),
        ci_high=min(1.0, mean + 2 * std),
        is_uncertain=(std >= 0.15),
        escalation_required=(std >= 0.15 or mean < 0.60),
        n_passes=20,
    )


@pytest.fixture
def pipeline():
    """Return a pipeline with no models loaded — for logic-only tests."""
    return EdgeInferencePipeline()


# ------------------------------------------------------------------ #
# Tests — core verdict logic (no model inference needed)
# ------------------------------------------------------------------ #

class TestVerdictLogic:

    def test_no_detections_returns_pass(self, pipeline):
        """Empty detections list → PASS."""
        verdict, escalated = pipeline._apply_verdict_logic([], uq=None)
        assert verdict == Verdict.PASS
        assert escalated is False

    def test_high_confidence_certain_returns_fail(self, pipeline):
        """conf ≥ CONFIRMED_DEFECT_THRESHOLD (0.85) and not uncertain → FAIL."""
        dets = [_make_detection(confidence=0.90)]
        uq = _make_uq(mean=0.90, std=0.02)  # certain (std < 0.15)
        verdict, escalated = pipeline._apply_verdict_logic(dets, uq=uq)
        assert verdict == Verdict.FAIL
        assert escalated is False

    def test_confident_but_uncertain_escalates(self, pipeline):
        """conf ≥ ESCALATE_THRESHOLD (0.60) but uncertain → FAIL with escalated=True."""
        dets = [_make_detection(confidence=0.75)]
        uq = _make_uq(mean=0.75, std=0.20)  # uncertain (std ≥ 0.15)
        verdict, escalated = pipeline._apply_verdict_logic(dets, uq=uq)
        # mean=0.75 < 0.85 so first branch fails; 0.75 ≥ 0.60 → second branch
        assert verdict == Verdict.FAIL
        assert escalated is True

    def test_medium_confidence_returns_escalate(self, pipeline):
        """conf in [HUMAN_REVIEW_THRESHOLD, ESCALATE_THRESHOLD) → ESCALATE."""
        dets = [_make_detection(confidence=0.50)]
        uq = _make_uq(mean=0.50, std=0.05)
        verdict, escalated = pipeline._apply_verdict_logic(dets, uq=uq)
        assert verdict == Verdict.ESCALATE
        assert escalated is True

    def test_low_confidence_returns_review(self, pipeline):
        """conf < HUMAN_REVIEW_THRESHOLD (0.45) → REVIEW."""
        dets = [_make_detection(confidence=0.30)]
        uq = _make_uq(mean=0.30, std=0.05)
        verdict, escalated = pipeline._apply_verdict_logic(dets, uq=uq)
        assert verdict == Verdict.REVIEW
        assert escalated is True

    def test_fail_not_pass_when_defect_present(self, pipeline):
        """Regression — original bug had PASS/FAIL inverted. Ensure a high-conf
        defect never returns PASS."""
        for conf in (0.86, 0.90, 0.95, 1.0):
            dets = [_make_detection(confidence=conf)]
            uq = _make_uq(mean=conf, std=0.01)
            verdict, _ = pipeline._apply_verdict_logic(dets, uq=uq)
            assert verdict != Verdict.PASS, (
                f"PASS returned for conf={conf} — inverted verdict bug!"
            )

    def test_verdict_without_uq_uses_detection_confidence(self, pipeline):
        """When uq=None, use max detection confidence directly."""
        dets = [_make_detection(confidence=0.90)]
        verdict, escalated = pipeline._apply_verdict_logic(dets, uq=None)
        # conf=0.90 ≥ 0.85 and no uncertainty info → FAIL (not uncertain by default)
        assert verdict == Verdict.FAIL
        assert escalated is False

    def test_multiple_detections_uses_uq_mean(self, pipeline):
        """With multiple detections, UQ mean governs the verdict."""
        dets = [
            _make_detection(confidence=0.95),
            _make_detection(confidence=0.50),
        ]
        uq = _make_uq(mean=0.88, std=0.03)
        verdict, _ = pipeline._apply_verdict_logic(dets, uq=uq)
        assert verdict == Verdict.FAIL
