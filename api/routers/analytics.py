"""
api/routers/analytics.py — Analytics and reporting endpoints.

Routes:
  GET  /analytics/summary              — KPI summary for current window
  GET  /analytics/defect-rate          — Defect rate over time
  GET  /analytics/defect-pareto        — Defect count by class (Pareto)
  GET  /analytics/severity-distribution — Count per severity grade
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from api.dependencies import get_db, verify_api_key
from database.repositories.inspection_repository import InspectionRepository

router = APIRouter(prefix="/analytics", tags=["Analytics"])


@router.get(
    "/summary",
    dependencies=[Depends(verify_api_key)],
    summary="KPI dashboard summary",
)
def get_summary(
    hours: int = Query(24, ge=1, le=720, description="Lookback window in hours"),
    db: Session = Depends(get_db),
) -> dict:
    """
    Returns aggregated KPIs for the specified time window:
    - total_inspections
    - by_verdict breakdown
    - overall defect_rate
    - avg_latency_ms
    """
    repo = InspectionRepository(db)
    return repo.summary(hours=hours)


@router.get(
    "/defect-pareto",
    dependencies=[Depends(verify_api_key)],
    summary="Defect count by class (Pareto)",
)
def get_pareto(
    hours: int = Query(24, ge=1, le=720),
    db: Session = Depends(get_db),
) -> list[dict]:
    """
    Returns per-class defect counts sorted descending (Pareto order).
    Use this to build the Pareto chart in the QA dashboard.
    """
    repo = InspectionRepository(db)
    return repo.defect_rate_by_class(hours=hours)


@router.get(
    "/severity-distribution",
    dependencies=[Depends(verify_api_key)],
    summary="Count per severity grade",
)
def get_severity_distribution(
    hours: int = Query(24, ge=1, le=720),
    db: Session = Depends(get_db),
) -> list[dict]:
    """Returns count of S1/S2/S3/S4 defects in the given window."""
    repo = InspectionRepository(db)
    return repo.severity_distribution(hours=hours)
