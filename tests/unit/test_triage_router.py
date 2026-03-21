"""
tests/unit/test_triage_router.py — Unit tests for TriageRouter.

Covers: routing table correctness, max-attempt override, reject paths, S3/S4.
"""

from __future__ import annotations

import pytest

from core.schemas import (
    BoundingBox,
    DefectClass,
    Detection,
    RemediationActionType,
    SeverityGrade,
    SeverityResult,
)
from remedy.triage_router import TriageRouter


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

def _make_detection(class_name: DefectClass) -> Detection:
    bbox = BoundingBox(x1=0.1, y1=0.1, x2=0.4, y2=0.4)
    return Detection(
        class_id=0,
        class_name=class_name,
        confidence=0.90,
        bbox=bbox,
        bbox_area_ratio=bbox.area_ratio,
    )


def _make_severity(grade: SeverityGrade, score: float = 0.40) -> SeverityResult:
    return SeverityResult(
        grade=grade,
        score=score,
        area_component=0.10,
        conf_uncertainty_component=0.05,
        class_risk_component=0.20,
        attempt_penalty_component=0.05,
    )


@pytest.fixture
def router():
    return TriageRouter()


# ------------------------------------------------------------------ #
# Tests — routing table
# ------------------------------------------------------------------ #

class TestRoutingTable:

    def test_label_misalignment_s1_to_relabel(self, router):
        det = _make_detection(DefectClass.LABEL_MISALIGNMENT)
        sev = _make_severity(SeverityGrade.S1)
        action = router.route(det, sev, attempt_count=0)
        assert action.action == RemediationActionType.RELABEL
        assert action.station == "A"
        assert action.is_remediable is True

    def test_label_misalignment_s2_to_relabel(self, router):
        det = _make_detection(DefectClass.LABEL_MISALIGNMENT)
        sev = _make_severity(SeverityGrade.S2)
        action = router.route(det, sev, attempt_count=0)
        assert action.action == RemediationActionType.RELABEL

    def test_improper_filling_s1_to_refill_station_b(self, router):
        det = _make_detection(DefectClass.IMPROPER_FILLING)
        sev = _make_severity(SeverityGrade.S1)
        action = router.route(det, sev, attempt_count=0)
        assert action.action == RemediationActionType.REFILL
        assert action.station == "B"

    def test_packaging_damage_s2_to_repack_station_c(self, router):
        det = _make_detection(DefectClass.PACKAGING_DAMAGE)
        sev = _make_severity(SeverityGrade.S2)
        action = router.route(det, sev, attempt_count=0)
        assert action.action == RemediationActionType.REPACK
        assert action.station == "C"

    def test_surface_contamination_s1_to_clean(self, router):
        det = _make_detection(DefectClass.SURFACE_CONTAMINATION)
        sev = _make_severity(SeverityGrade.S1)
        action = router.route(det, sev, attempt_count=0)
        assert action.action == RemediationActionType.CLEAN
        assert action.station == "C"


class TestRejectPaths:

    def test_any_class_s3_to_reject(self, router):
        for cls in DefectClass:
            det = _make_detection(cls)
            sev = _make_severity(SeverityGrade.S3, score=0.70)
            action = router.route(det, sev, attempt_count=0)
            assert action.action == RemediationActionType.REJECT, (
                f"Expected REJECT for {cls} S3, got {action.action}"
            )
            assert action.is_remediable is False

    def test_any_class_s4_to_reject(self, router):
        for cls in DefectClass:
            det = _make_detection(cls)
            sev = _make_severity(SeverityGrade.S4, score=0.90)
            action = router.route(det, sev, attempt_count=0)
            assert action.action == RemediationActionType.REJECT
            assert action.is_remediable is False

    def test_surface_contamination_s2_to_reject(self, router):
        """Surface contamination above S1 must always reject."""
        det = _make_detection(DefectClass.SURFACE_CONTAMINATION)
        sev = _make_severity(SeverityGrade.S2)
        action = router.route(det, sev, attempt_count=0)
        assert action.action == RemediationActionType.REJECT

    def test_max_attempts_forces_reject(self, router):
        """Even a remediable defect must REJECT if attempt_count ≥ MAX_ATTEMPTS."""
        det = _make_detection(DefectClass.LABEL_MISALIGNMENT)
        sev = _make_severity(SeverityGrade.S1)
        action = router.route(det, sev, attempt_count=2)
        assert action.action == RemediationActionType.REJECT
        assert action.is_remediable is False

    def test_attempt_count_below_max_still_remediable(self, router):
        """attempt_count=1 (< MAX_ATTEMPTS=2) must still be routed normally."""
        det = _make_detection(DefectClass.IMPROPER_FILLING)
        sev = _make_severity(SeverityGrade.S1)
        action = router.route(det, sev, attempt_count=1)
        assert action.action == RemediationActionType.REFILL
        assert action.is_remediable is True

    def test_reject_action_has_no_station(self, router):
        """REJECT actions must not assign a station."""
        det = _make_detection(DefectClass.PACKAGING_DAMAGE)
        sev = _make_severity(SeverityGrade.S4, score=0.92)
        action = router.route(det, sev, attempt_count=0)
        assert action.action == RemediationActionType.REJECT
        assert action.station is None
