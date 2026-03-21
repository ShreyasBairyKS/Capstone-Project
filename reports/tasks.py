"""
reports/tasks.py — Celery tasks for async report generation.

These tasks are discovered by Celery via the `include` list in celery_app.py.
"""

from __future__ import annotations

from datetime import datetime

from celery import shared_task

from core.logging import get_logger
from database.session import get_db_session
from database.models import QualityReport
from database.repositories.inspection_repository import InspectionRepository
from reports.generator import ReportGenerator

log = get_logger(__name__)


@shared_task(name="reports.tasks.generate_pdf", bind=True, max_retries=3)
def generate_pdf(
    self,
    report_id: int,
    from_dt: str,
    to_dt: str,
    sku: str | None = None,
) -> dict:
    """
    Celery task: generate a PDF quality report and update the DB record.

    Args:
        report_id: ID of the QualityReport row to update
        from_dt:   ISO-format start datetime
        to_dt:     ISO-format end datetime
        sku:       Optional SKU filter
    """
    from_dt_obj = datetime.fromisoformat(from_dt)
    to_dt_obj = datetime.fromisoformat(to_dt)

    db = get_db_session()
    try:
        report = db.get(QualityReport, report_id)
        if report is None:
            log.error("generate_pdf_report_not_found", report_id=report_id)
            return {"error": "Report not found"}

        # Mark as generating
        report.status = "generating"
        db.commit()

        # Gather analytics data
        repo = InspectionRepository(db)
        hours = max(1, int((to_dt_obj - from_dt_obj).total_seconds() / 3600))
        summary = repo.summary(hours=hours)
        pareto = repo.defect_rate_by_class(hours=hours)
        severity_dist = repo.severity_distribution(hours=hours)

        # Generate PDF
        generator = ReportGenerator()
        pdf_path = generator.generate(
            report_id=report_id,
            title=report.title,
            from_dt=from_dt_obj,
            to_dt=to_dt_obj,
            summary=summary,
            pareto=pareto,
            severity_dist=severity_dist,
            generated_by=report.generated_by,
        )

        # Update report record as complete
        report.status = "complete"
        report.pdf_path = str(pdf_path)
        report.total_inspections = summary.get("total_inspections", 0)
        report.defect_rate = summary.get("defect_rate", 0.0)
        report.completed_at = datetime.utcnow()
        db.commit()

        log.info("generate_pdf_complete", report_id=report_id, path=str(pdf_path))
        return {"report_id": report_id, "status": "complete", "pdf_path": str(pdf_path)}

    except Exception as exc:
        db.rollback()
        # Update DB to failed state
        try:
            report = db.get(QualityReport, report_id)
            if report:
                report.status = "failed"
                db.commit()
        except Exception:
            pass
        log.error("generate_pdf_failed", report_id=report_id, error=str(exc))
        raise self.retry(exc=exc, countdown=10)

    finally:
        db.close()
