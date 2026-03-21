"""
database/repositories/inspection_repository.py

Data access layer for Inspections, Defects, and Remediation Actions.
Provides CRUD operations and analytics aggregation queries.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from core.schemas import InspectionResult, Verdict
from database.models import Defect, Inspection, RemediationAction


class InspectionRepository:
    """CRUD + analytics queries for the inspections domain."""

    def __init__(self, db: Session) -> None:
        self.db = db

    # ------------------------------------------------------------------ #
    # Write operations
    # ------------------------------------------------------------------ #

    def save(self, result: InspectionResult, attempt_count: int = 0) -> Inspection:
        """
        Persist a complete InspectionResult to the database.

        Inserts the parent Inspection, associated Defect rows, and a
        RemediationAction row if one exists.

        Returns the persisted Inspection ORM object.
        """
        row = Inspection(
            id=result.inspection_id,
            product_id=result.product_id,
            sku=result.sku,
            timestamp=result.timestamp,
            verdict=result.verdict.value,
            escalated=result.escalated,
            latency_ms=result.latency_ms,
            device_id=result.device_id,
            attempt_count=attempt_count,
        )
        self.db.add(row)

        # Persist each detected defect
        for d in result.detections:
            uq_mean = result.uq_result.mean_confidence if result.uq_result else None
            uq_std = result.uq_result.std_confidence if result.uq_result else None
            severity_grade = (
                result.severity_result.grade.value if result.severity_result else None
            )
            severity_score = (
                result.severity_result.score if result.severity_result else None
            )
            defect_row = Defect(
                inspection_id=result.inspection_id,
                class_name=d.class_name.value,
                confidence=d.confidence,
                bbox_x1=d.bbox.x1,
                bbox_y1=d.bbox.y1,
                bbox_x2=d.bbox.x2,
                bbox_y2=d.bbox.y2,
                bbox_area_ratio=d.bbox_area_ratio,
                severity_grade=severity_grade,
                severity_score=severity_score,
                uq_mean=uq_mean,
                uq_std=uq_std,
            )
            self.db.add(defect_row)

        # Persist remediation action if present
        if result.remediation_action:
            ra = RemediationAction(
                inspection_id=result.inspection_id,
                action=result.remediation_action.action.value,
                station=result.remediation_action.station,
                is_remediable=result.remediation_action.is_remediable,
                reason=result.remediation_action.reason,
            )
            self.db.add(ra)

        self.db.commit()
        self.db.refresh(row)
        return row

    def mark_remediation_complete(self, inspection_id: str) -> bool:
        """
        Mark a remediation action as completed. Returns True if found and updated.
        """
        stmt = select(RemediationAction).where(
            RemediationAction.inspection_id == inspection_id
        )
        action = self.db.execute(stmt).scalar_one_or_none()
        if action is None:
            return False
        action.completed = True
        action.completed_at = datetime.utcnow()
        self.db.commit()
        return True

    def override_verdict(
        self, inspection_id: str, new_verdict: str, reason: str
    ) -> Optional[Inspection]:
        """
        Operator override — update verdict and record reason in a note.
        Returns None if inspection not found.
        """
        stmt = select(Inspection).where(Inspection.id == inspection_id)
        row = self.db.execute(stmt).scalar_one_or_none()
        if row is None:
            return None
        row.verdict = new_verdict
        self.db.commit()
        self.db.refresh(row)
        return row

    # ------------------------------------------------------------------ #
    # Read operations
    # ------------------------------------------------------------------ #

    def get_by_id(self, inspection_id: str) -> Optional[Inspection]:
        stmt = select(Inspection).where(Inspection.id == inspection_id)
        return self.db.execute(stmt).scalar_one_or_none()

    def list_recent(
        self,
        limit: int = 50,
        offset: int = 0,
        verdict: Optional[str] = None,
        sku: Optional[str] = None,
        from_dt: Optional[datetime] = None,
        to_dt: Optional[datetime] = None,
    ) -> list[Inspection]:
        """Paginated list of inspections with optional filters."""
        stmt = select(Inspection).order_by(Inspection.timestamp.desc())
        if verdict:
            stmt = stmt.where(Inspection.verdict == verdict)
        if sku:
            stmt = stmt.where(Inspection.sku == sku)
        if from_dt:
            stmt = stmt.where(Inspection.timestamp >= from_dt)
        if to_dt:
            stmt = stmt.where(Inspection.timestamp <= to_dt)
        stmt = stmt.offset(offset).limit(limit)
        return list(self.db.execute(stmt).scalars().all())

    def count_total(
        self,
        from_dt: Optional[datetime] = None,
        to_dt: Optional[datetime] = None,
    ) -> int:
        stmt = select(func.count(Inspection.id))
        if from_dt:
            stmt = stmt.where(Inspection.timestamp >= from_dt)
        if to_dt:
            stmt = stmt.where(Inspection.timestamp <= to_dt)
        return self.db.execute(stmt).scalar() or 0

    # ------------------------------------------------------------------ #
    # Analytics queries
    # ------------------------------------------------------------------ #

    def summary(
        self,
        hours: int = 24,
    ) -> dict:
        """
        Returns a summary dict for the analytics dashboard.
        Covers the last `hours` hours.
        """
        since = datetime.utcnow() - timedelta(hours=hours)

        total = self.db.execute(
            select(func.count(Inspection.id)).where(Inspection.timestamp >= since)
        ).scalar() or 0

        by_verdict = dict(
            self.db.execute(
                select(Inspection.verdict, func.count(Inspection.id))
                .where(Inspection.timestamp >= since)
                .group_by(Inspection.verdict)
            ).all()
        )

        fail_count = by_verdict.get(Verdict.FAIL.value, 0)
        defect_rate = round(fail_count / total, 4) if total > 0 else 0.0

        avg_latency = self.db.execute(
            select(func.avg(Inspection.latency_ms)).where(
                Inspection.timestamp >= since
            )
        ).scalar() or 0.0

        return {
            "period_hours": hours,
            "total_inspections": total,
            "by_verdict": by_verdict,
            "defect_rate": defect_rate,
            "avg_latency_ms": round(float(avg_latency), 2),
        }

    def defect_rate_by_class(self, hours: int = 24) -> list[dict]:
        """Per-class defect counts for the pareto chart."""
        since = datetime.utcnow() - timedelta(hours=hours)

        rows = self.db.execute(
            select(Defect.class_name, func.count(Defect.id))
            .join(Inspection, Defect.inspection_id == Inspection.id)
            .where(Inspection.timestamp >= since)
            .group_by(Defect.class_name)
            .order_by(func.count(Defect.id).desc())
        ).all()

        return [{"class_name": r[0], "count": r[1]} for r in rows]

    def severity_distribution(self, hours: int = 24) -> list[dict]:
        """Count of each severity grade over the window."""
        since = datetime.utcnow() - timedelta(hours=hours)

        rows = self.db.execute(
            select(Defect.severity_grade, func.count(Defect.id))
            .join(Inspection, Defect.inspection_id == Inspection.id)
            .where(Inspection.timestamp >= since)
            .where(Defect.severity_grade.isnot(None))
            .group_by(Defect.severity_grade)
        ).all()

        return [{"grade": r[0], "count": r[1]} for r in rows]
