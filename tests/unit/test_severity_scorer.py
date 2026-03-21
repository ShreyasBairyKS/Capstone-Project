"""
tests/unit/test_severity_scorer.py — Unit tests for SeverityScorer.

Covers: scoring formula, grade thresholds, attempt penalty, edge cases.
"""

from __future__ import annotations

import pytest

from core.schemas import (
    BoundingBox,
    DefectClass,
    Detection,
    SeverityGrade,
    UQResult,
)
from remedy.severity_scorer import SeverityScorer


# ------------------------------------------------------------------ #
# Fixtures
# ------------------------------------------------------------------ #

def _make_detection(
    class_name: DefectClass = DefectClass.LABEL_MISALIGNMENT,
    confidence: float = 0.90,
    x1: float = 0.1,
    y1: float = 0.1,
    x2: float = 0.3,
    y2: float = 0.3,
) -> Detection:
    bbox = BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)
    return Detection(
        class_id=0,
        class_name=class_name,
        confidence=confidence,
        bbox=bbox,
        bbox_area_ratio=bbox.area_ratio,
    )


def _make_uq(mean: float = 0.90, std: float = 0.05) -> UQResult:
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
def scorer():
    return SeverityScorer()


# ------------------------------------------------------------------ #
# Tests
# ------------------------------------------------------------------ #

class TestScorerGrades:

    def test_s1_low_risk_small_area(self, scorer):
        """Label misalignment, tiny bbox → S1."""
        det = _make_detection(
            class_name=DefectClass.LABEL_MISALIGNMENT,
            x1=0.4, y1=0.4, x2=0.5, y2=0.5,  # 1% area
        )
        result = scorer.score(det, uq=None, attempt_count=0)
        assert result.grade == SeverityGrade.S1
        assert 0.0 <= result.score < 0.30

    def test_s2_medium_risk(self, scorer):
        """Improper filling, moderate area, some uncertainty → S2."""
        det = _make_detection(
            class_name=DefectClass.IMPROPER_FILLING,
            x1=0.2, y1=0.2, x2=0.5, y2=0.5,   # 9% area
        )
        uq = _make_uq(mean=0.70, std=0.08)
        result = scorer.score(det, uq=uq, attempt_count=0)
        assert result.grade in (SeverityGrade.S1, SeverityGrade.S2)

    def test_s4_surface_contamination_large_area(self, scorer):
        """Surface contamination (highest risk class), large bbox → S4."""
        det = _make_detection(
            class_name=DefectClass.SURFACE_CONTAMINATION,
            x1=0.0, y1=0.0, x2=0.8, y2=0.8,   # 64% area → capped at 25%
        )
        uq = _make_uq(mean=0.55, std=0.20)
        result = scorer.score(det, uq=uq, attempt_count=2)
        assert result.grade in (SeverityGrade.S3, SeverityGrade.S4)
        assert result.score >= 0.55

    def test_score_in_bounds(self, scorer):
        """Score must always be in [0.0, 1.0]."""
        for cls in DefectClass:
            det = _make_detection(class_name=cls, x1=0.0, y1=0.0, x2=1.0, y2=1.0)
            uq = _make_uq(mean=0.50, std=0.30)
            result = scorer.score(det, uq=uq, attempt_count=5)
            assert 0.0 <= result.score <= 1.0, f"Out-of-bounds score for {cls}"

    def test_attempt_penalty_increases_score(self, scorer):
        """More remediation attempts → higher score."""
        det = _make_detection(class_name=DefectClass.PACKAGING_DAMAGE)
        r0 = scorer.score(det, uq=None, attempt_count=0)
        r2 = scorer.score(det, uq=None, attempt_count=2)
        assert r2.score >= r0.score

    def test_uq_uncertainty_increases_score(self, scorer):
        """Higher std_confidence → higher score contribution."""
        det = _make_detection(class_name=DefectClass.PACKAGING_DAMAGE)
        r_low_uq = scorer.score(det, uq=_make_uq(std=0.02), attempt_count=0)
        r_high_uq = scorer.score(det, uq=_make_uq(std=0.25), attempt_count=0)
        assert r_high_uq.score >= r_low_uq.score

    def test_no_uq_does_not_crash(self, scorer):
        """Passing uq=None must not raise."""
        det = _make_detection()
        result = scorer.score(det, uq=None, attempt_count=0)
        assert result.grade is not None
        assert result.conf_uncertainty_component == 0.0

    def test_component_breakdown_sums_correctly(self, scorer):
        """Component weights sum to 1.0 for the formula."""
        det = _make_detection(
            class_name=DefectClass.IMPROPER_FILLING,
            x1=0.1, y1=0.1, x2=0.4, y2=0.4,
        )
        uq = _make_uq(mean=0.75, std=0.10)
        result = scorer.score(det, uq=uq, attempt_count=1)
        # Verify all components are non-negative
        assert result.area_component >= 0.0
        assert result.conf_uncertainty_component >= 0.0
        assert result.class_risk_component >= 0.0
        assert result.attempt_penalty_component >= 0.0
