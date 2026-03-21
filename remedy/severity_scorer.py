"""
remedy/severity_scorer.py

REMEDY Severity Scorer — assigns S1/S2/S3/S4 severity grade to a detection.

Severity formula (weighted sum, normalised to [0, 1]):
    score = 0.35 × area_ratio
          + 0.15 × conf_uncertainty
          + 0.40 × class_risk
          + 0.10 × attempt_penalty

Grade thresholds:
    S1: score < 0.30  (minor, remediable on-line)
    S2: score < 0.55  (moderate, station remediation)
    S3: score < 0.80  (severe, reject)
    S4: score ≥ 0.80  (critical, quarantine)

Implemented in Phase 3.
"""

from __future__ import annotations

from typing import Optional

from core.schemas import (
    DefectClass,
    Detection,
    SeverityGrade,
    SeverityResult,
    UQResult,
)

# Class risk weights — based on food safety impact
CLASS_RISK: dict[DefectClass, float] = {
    DefectClass.SURFACE_CONTAMINATION: 1.00,   # Highest food safety risk
    DefectClass.IMPROPER_FILLING: 0.75,
    DefectClass.PACKAGING_DAMAGE: 0.65,
    DefectClass.LABEL_MISALIGNMENT: 0.30,      # Cosmetic, lowest risk
}

# Score → grade thresholds
GRADE_THRESHOLDS = [
    (0.30, SeverityGrade.S1),
    (0.55, SeverityGrade.S2),
    (0.80, SeverityGrade.S3),
]


class SeverityScorer:
    """
    Computes REMEDY severity score for a single defect detection.

    Usage (Phase 3):
        scorer = SeverityScorer()
        result = scorer.score(detection, uq_result, attempt_count=0)
    """

    def score(
        self,
        detection: Detection,
        uq: Optional[UQResult],
        attempt_count: int = 0,
    ) -> SeverityResult:
        """
        Compute weighted severity score and assign grade.

        Args:
            detection:     Primary defect detection from YOLOv11 pipeline
            uq:            UQ result (used for conf_uncertainty component)
            attempt_count: Number of previous remediation attempts (penalty)

        Returns:
            SeverityResult with grade and component breakdown
        """
        area_component = min(detection.bbox_area_ratio / 0.25, 1.0)  # Capped at 25% area

        conf_uncertainty = uq.std_confidence if uq else 0.0
        conf_uncertainty_component = min(conf_uncertainty / 0.30, 1.0)

        class_risk_component = CLASS_RISK.get(detection.class_name, 0.5)

        attempt_penalty_component = min(attempt_count * 0.5, 1.0)

        score = (
            0.35 * area_component
            + 0.15 * conf_uncertainty_component
            + 0.40 * class_risk_component
            + 0.10 * attempt_penalty_component
        )
        score = round(min(score, 1.0), 4)
        grade = self._assign_grade(score)

        return SeverityResult(
            grade=grade,
            score=score,
            area_component=round(area_component, 4),
            conf_uncertainty_component=round(conf_uncertainty_component, 4),
            class_risk_component=round(class_risk_component, 4),
            attempt_penalty_component=round(attempt_penalty_component, 4),
        )

    @staticmethod
    def _assign_grade(score: float) -> SeverityGrade:
        for threshold, grade in GRADE_THRESHOLDS:
            if score < threshold:
                return grade
        return SeverityGrade.S4
