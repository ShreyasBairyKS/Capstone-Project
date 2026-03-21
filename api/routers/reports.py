"""
api/routers/reports.py — Quality report generation and download.

Routes:
  POST   /reports/generate          — Trigger async PDF report generation
  GET    /reports/{report_id}/status — Poll generation status
  GET    /reports/{report_id}/download — Download the generated PDF
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from api.dependencies import get_db, verify_api_key
from database.models import QualityReport

router = APIRouter(prefix="/reports", tags=["Reports"])

_REPORTS_DIR = Path("reports/output")
_REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------------ #
# Request / internal schemas
# ------------------------------------------------------------------ #

class ReportGenerateRequest(BaseModel):
    title: str = Field("VisionFood QAI Quality Report", max_length=200)
    from_dt: datetime
    to_dt: datetime
    sku: Optional[str] = Field(None, max_length=64)
    generated_by: str = Field("system", max_length=128)


def _row_to_dict(r: QualityReport) -> dict:
    return {
        "id": r.id,
        "title": r.title,
        "generated_at": r.generated_at.isoformat() if r.generated_at else None,
        "from_dt": r.from_dt.isoformat() if r.from_dt else None,
        "to_dt": r.to_dt.isoformat() if r.to_dt else None,
        "total_inspections": r.total_inspections,
        "defect_rate": r.defect_rate,
        "pdf_path": r.pdf_path,
        "status": r.status,
        "generated_by": r.generated_by,
    }


# ------------------------------------------------------------------ #
# Endpoints
# ------------------------------------------------------------------ #

@router.post(
    "/generate",
    response_model=dict,
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[Depends(verify_api_key)],
    summary="Trigger PDF report generation",
)
def generate_report(
    body: ReportGenerateRequest,
    db: Session = Depends(get_db),
) -> dict:
    """
    Enqueues a Celery task to generate a PDF quality report.
    Returns the report ID immediately; poll /reports/{id}/status for progress.
    """
    from api.celery_app import celery_app

    # Create a pending DB record
    report = QualityReport(
        title=body.title,
        from_dt=body.from_dt,
        to_dt=body.to_dt,
        generated_at=datetime.utcnow(),
        status="pending",
        generated_by=body.generated_by,
        total_inspections=0,
        defect_rate=0.0,
    )
    db.add(report)
    db.commit()
    db.refresh(report)

    # Dispatch async generation task
    celery_app.send_task(
        "reports.tasks.generate_pdf",
        kwargs={
            "report_id": report.id,
            "from_dt": body.from_dt.isoformat(),
            "to_dt": body.to_dt.isoformat(),
            "sku": body.sku,
        },
    )

    return {"report_id": report.id, "status": "pending"}


@router.get(
    "/{report_id}/status",
    response_model=dict,
    dependencies=[Depends(verify_api_key)],
    summary="Poll report generation status",
)
def report_status(
    report_id: int,
    db: Session = Depends(get_db),
) -> dict:
    report = db.get(QualityReport, report_id)
    if report is None:
        raise HTTPException(status_code=404, detail="Report not found.")
    return _row_to_dict(report)


@router.get(
    "/{report_id}/download",
    dependencies=[Depends(verify_api_key)],
    summary="Download generated PDF",
)
def download_report(
    report_id: int,
    db: Session = Depends(get_db),
) -> FileResponse:
    report = db.get(QualityReport, report_id)
    if report is None:
        raise HTTPException(status_code=404, detail="Report not found.")
    if report.status != "complete":
        raise HTTPException(
            status_code=409,
            detail=f"Report is not ready yet (status: {report.status}).",
        )
    pdf_path = Path(report.pdf_path)
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail="PDF file not found on disk.")
    return FileResponse(
        path=str(pdf_path),
        media_type="application/pdf",
        filename=pdf_path.name,
    )
