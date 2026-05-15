"""
remedy/severity_scorer.py

REMEDY Severity Scorer — assigns S1/S2/S3/S4 severity grade to a detection.

Severity formula (weighted sum, normalised to [0, 1]):
    score = w_area     × area_ratio
          + w_conf_uq  × conf_uncertainty
          + w_class    × class_risk
          + w_attempt  × attempt_penalty

All weights, grade thresholds, and normalisation caps are driven by EdgeConfig
so tuning requires zero code changes — only environment variables or .env edits.
"""

from __future__ import annotations

from typing import Optional

from core.config import EdgeConfig, settings
from core.schemas import (
    DefectClass,
    Detection,
    SeverityGrade,
    SeverityResult,
    UQResult,
)

# Class risk weights — based on food safety impact (overridable via SKU profiles)
DEFAULT_CLASS_RISK: dict[DefectClass, float] = {
    DefectClass.SURFACE_CONTAMINATION: 1.00,
    DefectClass.IMPROPER_FILLING: 0.75,
    DefectClass.PACKAGING_DAMAGE: 0.65,
    DefectClass.LABEL_MISALIGNMENT: 0.30,
    DefectClass.FILL_LEVEL_LOW: 0.80,
    DefectClass.FILL_LEVEL_HIGH: 0.80,
    DefectClass.CAP_FITTING_ANOMALY: 0.65,
    DefectClass.SURFACE_TEAR: 0.70,
    DefectClass.SURFACE_SMUDGE: 0.45,
    DefectClass.LABEL_DATE_MISMATCH: 0.70,
    DefectClass.LABEL_BARCODE_MISMATCH: 0.90,
}


class SeverityScorer:
    """
    Computes REMEDY severity score for a single defect detection.

    All parameters are read from the supplied EdgeConfig at construction time,
    making the scorer fully configurable without code changes.
    """

    def __init__(
        self,
        config: Optional[EdgeConfig] = None,
        class_risk_overrides: Optional[dict[DefectClass, float]] = None,
    ) -> None:
        cfg = config or settings
        self._w_area = cfg.SEVERITY_W_AREA
        self._w_conf_uq = cfg.SEVERITY_W_CONF_UQ
        self._w_class_risk = cfg.SEVERITY_W_CLASS_RISK
        self._w_attempt = cfg.SEVERITY_W_ATTEMPT
        self._area_cap = cfg.SEVERITY_AREA_CAP
        self._conf_uq_cap = cfg.SEVERITY_CONF_UQ_CAP
        self._grade_thresholds = [
            (cfg.SEVERITY_THRESHOLD_S1, SeverityGrade.S1),
            (cfg.SEVERITY_THRESHOLD_S2, SeverityGrade.S2),
            (cfg.SEVERITY_THRESHOLD_S3, SeverityGrade.S3),
        ]
        self._class_risk = class_risk_overrides if class_risk_overrides else dict(DEFAULT_CLASS_RISK)

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
        area_component = min(detection.bbox_area_ratio / self._area_cap, 1.0)

        conf_uncertainty = uq.std_confidence if uq else 0.0
        conf_uncertainty_component = min(conf_uncertainty / self._conf_uq_cap, 1.0)

        class_risk_component = self._class_risk.get(detection.class_name, 0.5)

        attempt_penalty_component = min(attempt_count * 0.5, 1.0)

        score = (
            self._w_area * area_component
            + self._w_conf_uq * conf_uncertainty_component
            + self._w_class_risk * class_risk_component
            + self._w_attempt * attempt_penalty_component
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

    def _assign_grade(self, score: float) -> SeverityGrade:
        for threshold, grade in self._grade_thresholds:
            if score < threshold:
                return grade
        return SeverityGrade.S4
