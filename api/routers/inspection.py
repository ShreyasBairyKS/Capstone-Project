"""
api/routers/inspection.py — Inspection endpoints.

Routes:
  POST   /inspections          — Submit an image for QA inspection
  GET    /inspections          — List recent inspections (paginated + filtered)
  GET    /inspections/{id}     — Get a single inspection by ID
  PATCH  /inspections/{id}/verdict — Operator override verdict
"""

from __future__ import annotations

import base64
from datetime import datetime
from typing import Optional

import cv2
import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.orm import Session

from api.dependencies import get_db, get_pipeline, verify_api_key
from core.schemas import InspectionResult
from database.repositories.inspection_repository import InspectionRepository
from inference.pipeline import EdgeInferencePipeline

router = APIRouter(prefix="/inspections", tags=["Inspections"])


# ------------------------------------------------------------------ #
# Request / Response schemas
# ------------------------------------------------------------------ #

class InspectRequest(BaseModel):
    """
    Image must be base64-encoded JPEG/PNG.
    product_id, sku, and attempt_count are optional metadata.
    """
    image_b64: str = Field(..., description="Base64-encoded image (JPEG or PNG)")
    product_id: Optional[str] = Field(None, max_length=64)
    sku: str = Field("default", max_length=64)
    attempt_count: int = Field(0, ge=0)

    @field_validator("image_b64")
    @classmethod
    def validate_image(cls, v: str) -> str:
        try:
            base64.b64decode(v, validate=True)
        except Exception:
            raise ValueError("image_b64 is not valid base64.")
        return v


class VerdictOverrideRequest(BaseModel):
    new_verdict: str = Field(..., pattern="^(PASS|FAIL|ESCALATE|REVIEW)$")
    reason: str = Field(..., min_length=5, max_length=500)


def _decode_image(image_b64: str) -> np.ndarray:
    """Decode a base64-encoded image string to a BGR numpy array."""
    try:
        raw = base64.b64decode(image_b64)
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="image_b64 is not valid base64.",
        )
    arr = np.frombuffer(raw, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Could not decode image. Ensure it is a valid JPEG or PNG.",
        )
    return frame


# ------------------------------------------------------------------ #
# Endpoints
# ------------------------------------------------------------------ #

@router.post(
    "",
    response_model=InspectionResult,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(verify_api_key)],
    summary="Submit product image for QA inspection",
)
async def create_inspection(
    body: InspectRequest,
    pipeline: EdgeInferencePipeline = Depends(get_pipeline),
    db: Session = Depends(get_db),
) -> InspectionResult:
    """
    Run the full VisionFood QAI inference pipeline on the submitted image.

    Returns a complete InspectionResult including:
    - Verdict (PASS / FAIL / ESCALATE / REVIEW)
    - All detected defects with bounding boxes
    - UQ confidence interval
    - REMEDY severity grade and remediation action
    """
    frame = _decode_image(body.image_b64)
    result = pipeline.inspect(
        frame=frame,
        product_id=body.product_id,
        sku=body.sku,
        attempt_count=body.attempt_count,
    )
    repo = InspectionRepository(db)
    repo.save(result, attempt_count=body.attempt_count)
    return result


@router.get(
    "",
    response_model=list[dict],
    dependencies=[Depends(verify_api_key)],
    summary="List recent inspections",
)
def list_inspections(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    verdict: Optional[str] = Query(None, pattern="^(PASS|FAIL|ESCALATE|REVIEW)$"),
    sku: Optional[str] = Query(None, max_length=64),
    from_dt: Optional[datetime] = Query(None),
    to_dt: Optional[datetime] = Query(None),
    db: Session = Depends(get_db),
) -> list[dict]:
    """Return paginated list of inspections, newest first."""
    repo = InspectionRepository(db)
    rows = repo.list_recent(
        limit=limit,
        offset=offset,
        verdict=verdict,
        sku=sku,
        from_dt=from_dt,
        to_dt=to_dt,
    )
    return [
        {
            "id": r.id,
            "product_id": r.product_id,
            "sku": r.sku,
            "timestamp": r.timestamp.isoformat(),
            "verdict": r.verdict,
            "escalated": r.escalated,
            "latency_ms": r.latency_ms,
            "device_id": r.device_id,
            "defect_count": len(r.defects),
        }
        for r in rows
    ]


@router.get(
    "/{inspection_id}",
    response_model=dict,
    dependencies=[Depends(verify_api_key)],
    summary="Get a single inspection",
)
def get_inspection(
    inspection_id: str,
    db: Session = Depends(get_db),
) -> dict:
    repo = InspectionRepository(db)
    row = repo.get_by_id(inspection_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Inspection not found.")
    return {
        "id": row.id,
        "product_id": row.product_id,
        "sku": row.sku,
        "timestamp": row.timestamp.isoformat(),
        "verdict": row.verdict,
        "escalated": row.escalated,
        "latency_ms": row.latency_ms,
        "device_id": row.device_id,
        "defects": [
            {
                "class_name": d.class_name,
                "confidence": d.confidence,
                "bbox": {
                    "x1": d.bbox_x1,
                    "y1": d.bbox_y1,
                    "x2": d.bbox_x2,
                    "y2": d.bbox_y2,
                },
                "severity_grade": d.severity_grade,
                "severity_score": d.severity_score,
            }
            for d in row.defects
        ],
        "remediation_action": (
            {
                "action": row.remediation_action.action,
                "station": row.remediation_action.station,
                "is_remediable": row.remediation_action.is_remediable,
                "reason": row.remediation_action.reason,
                "completed": row.remediation_action.completed,
            }
            if row.remediation_action
            else None
        ),
    }


@router.patch(
    "/{inspection_id}/verdict",
    response_model=dict,
    dependencies=[Depends(verify_api_key)],
    summary="Operator override on verdict",
)
def override_verdict(
    inspection_id: str,
    body: VerdictOverrideRequest,
    db: Session = Depends(get_db),
) -> dict:
    """Allow a human operator to correct the model's verdict."""
    repo = InspectionRepository(db)
    row = repo.override_verdict(inspection_id, body.new_verdict, body.reason)
    if row is None:
        raise HTTPException(status_code=404, detail="Inspection not found.")
    return {
        "id": row.id,
        "verdict": row.verdict,
        "message": f"Verdict updated to {row.verdict} by operator override.",
    }
