"""
remedy/triage_router.py

REMEDY Triage Router — maps (defect class, severity grade) to a remediation action.

Action → Station mapping:
    label_misalignment  S1/S2 → RELABEL  → Station A
    improper_filling    S1/S2 → REFILL   → Station B
    packaging_damage    S1/S2 → REPACK   → Station C
    surface_contamination S1  → CLEAN    → Station C
    anything S3/S4          → REJECT   → (no station)
    attempt_count ≥ max     → REJECT   regardless of grade
    no defects              → PASS
"""

from __future__ import annotations

from typing import Optional

from core.config import EdgeConfig, settings
from core.schemas import (
    DefectClass,
    Detection,
    RemediationAction,
    RemediationActionType,
    SeverityGrade,
    SeverityResult,
)

# Routing table: (class, grade) → (action, station, is_remediable)
_ROUTE_TABLE: dict[tuple[DefectClass, SeverityGrade], tuple[RemediationActionType, Optional[str], bool]] = {
    (DefectClass.LABEL_MISALIGNMENT, SeverityGrade.S1): (RemediationActionType.RELABEL, "A", True),
    (DefectClass.LABEL_MISALIGNMENT, SeverityGrade.S2): (RemediationActionType.RELABEL, "A", True),
    (DefectClass.IMPROPER_FILLING, SeverityGrade.S1):   (RemediationActionType.REFILL,  "B", True),
    (DefectClass.IMPROPER_FILLING, SeverityGrade.S2):   (RemediationActionType.REFILL,  "B", True),
    (DefectClass.PACKAGING_DAMAGE, SeverityGrade.S1):   (RemediationActionType.REPACK,  "C", True),
    (DefectClass.PACKAGING_DAMAGE, SeverityGrade.S2):   (RemediationActionType.REPACK,  "C", True),
    (DefectClass.SURFACE_CONTAMINATION, SeverityGrade.S1): (RemediationActionType.CLEAN, "C", True),
}


class TriageRouter:
    """
    Determines remediation routing for a detected defect.

    Max remediation attempts are read from EdgeConfig so production deployments
    can tune retry limits without code changes.
    """

    def __init__(self, config: Optional[EdgeConfig] = None) -> None:
        cfg = config or settings
        self._max_attempts = cfg.REMEDY_MAX_ATTEMPTS

    def route(
        self,
        detection: Detection,
        severity: SeverityResult,
        attempt_count: int = 0,
    ) -> RemediationAction:
        """
        Map detection + severity → remediation action.

        Args:
            detection:     Primary defect detection
            severity:      SeverityResult from SeverityScorer
            attempt_count: Previous remediation attempts for this product

        Returns:
            RemediationAction with action type, station, and reason
        """
        # Hard reject after max attempts regardless of grade
        if attempt_count >= self._max_attempts:
            return RemediationAction(
                action=RemediationActionType.REJECT,
                station=None,
                is_remediable=False,
                reason=f"Max remediation attempts ({self._max_attempts}) reached. Mandatory reject.",
                max_attempts=self._max_attempts,
            )

        # All S3/S4 → reject
        if severity.grade in (SeverityGrade.S3, SeverityGrade.S4):
            return RemediationAction(
                action=RemediationActionType.REJECT,
                station=None,
                is_remediable=False,
                reason=f"Severity {severity.grade.value} (score={severity.score:.3f}) exceeds remediation threshold.",
                max_attempts=self._max_attempts,
            )

        # Look up routing table
        key = (detection.class_name, severity.grade)
        if key in _ROUTE_TABLE:
            action_type, station, is_remediable = _ROUTE_TABLE[key]
            return RemediationAction(
                action=action_type,
                station=station,
                is_remediable=is_remediable,
                reason=(
                    f"{detection.class_name.value} defect at {severity.grade.value} severity "
                    f"(score={severity.score:.3f}). Route to Station {station}."
                ),
                max_attempts=self._max_attempts,
            )

        # Fallback — surface contamination S2/S3/S4 not in table → reject
        return RemediationAction(
            action=RemediationActionType.REJECT,
            station=None,
            is_remediable=False,
            reason=f"No remediation path for {detection.class_name.value} at {severity.grade.value}. Reject.",
            max_attempts=self._max_attempts,
        )
