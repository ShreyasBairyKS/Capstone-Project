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
import logging
from datetime import datetime
from typing import Literal, Optional

import cv2
import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query, status
from motor.motor_asyncio import AsyncIOMotorDatabase
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.orm import Session

from api.dependencies import get_db, get_motor_db, get_pipeline, verify_api_key
from core.schemas import InspectionResult
from database.repositories.inspection_repository import InspectionRepository
from database.repositories.production_run_repository import ProductionRunRepository
from database.repositories.product_repository import ProductRepository
from inference.pipeline import EdgeInferencePipeline

log = logging.getLogger(__name__)

router = APIRouter(prefix="/inspections", tags=["Inspections"])

PipelineMode = Literal["standard", "yolo_fill_level"]
_yolo_fill_pipeline = None


# ------------------------------------------------------------------ #
# Request / Response schemas
# ------------------------------------------------------------------ #

class InspectRequest(BaseModel):
    """
    Image must be base64-encoded JPEG/PNG.
    product_id, sku, and attempt_count are optional metadata.
    product_sub_type and container_contents are optional — if absent, they are
    resolved automatically from the active production run for the given SKU.
    """
    image_b64: str = Field(..., description="Base64-encoded image (JPEG or PNG)")
    product_id: Optional[str] = Field(None, max_length=64)
    sku: str = Field("default", max_length=64)
    attempt_count: int = Field(0, ge=0)
    pipeline_mode: Optional[PipelineMode] = Field(
        None,
        description="Inference backend: standard or yolo_fill_level. Defaults to live settings.",
    )
    use_cap_classifier: Optional[bool] = Field(
        None,
        description="When using yolo_fill_level, enable the cap quality classifier.",
    )
    product_category: Optional[str] = Field(
        None,
        max_length=32,
        description="Product category hint: beverage | food | general",
    )
    # Optional pipeline routing hints — resolved from active run when absent
    product_sub_type: Optional[str] = Field(
        None,
        description="Container sub-type hint: transparent_bottle | rigid_can | flexible_wrapper | rigid_box",
    )
    container_contents: Optional[str] = Field(
        None,
        description="Contents state hint: liquid | solid",
    )

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


class LiveInspectionSettings(BaseModel):
    pipeline_mode: PipelineMode = "standard"
    use_cap_classifier: bool = True


class LiveInspectionSettingsUpdate(BaseModel):
    pipeline_mode: Optional[PipelineMode] = None
    use_cap_classifier: Optional[bool] = None


_LIVE_SETTINGS = LiveInspectionSettings()


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


def _get_yolo_fill_pipeline():
    """Return the lazy singleton for inference/yoloWithFillLevel.py."""
    global _yolo_fill_pipeline
    if _yolo_fill_pipeline is None:
        from inference.yolo_fill_level_adapter import YoloFillLevelPipeline

        _yolo_fill_pipeline = YoloFillLevelPipeline()
    return _yolo_fill_pipeline


async def _publish_live_result(result: InspectionResult) -> None:
    """Best-effort Redis publication for the dashboard live WebSocket."""
    try:
        from core.messaging import get_redis_client, publish_inspection_event

        client = await get_redis_client()
        try:
            await publish_inspection_event(client, result.model_dump(mode="json"))
        finally:
            await client.aclose()
    except Exception as exc:  # noqa: BLE001 - live stream must not block inspection
        log.warning("live_publish_failed", extra={"error": str(exc)})


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
    motor_db: AsyncIOMotorDatabase = Depends(get_motor_db),
) -> InspectionResult:
    """
    Run the full VisionFood QAI inference pipeline on the submitted image.

    Returns a complete InspectionResult including:
    - Verdict (PASS / FAIL / ESCALATE / REVIEW)
    - All detected defects with bounding boxes
    - UQ confidence interval
    - REMEDY severity grade and remediation action

    ``product_sub_type`` and ``container_contents`` are resolved in this order:
    1. Values supplied directly in the request body.
    2. Active production run for the given SKU (looked up in MongoDB).
    3. None — the pipeline falls back to its default routing.
    """
    frame = _decode_image(body.image_b64)

    # --- Sub-type resolution ---
    product_category = body.product_category
    product_sub_type = body.product_sub_type
    container_contents = body.container_contents
    pipeline_mode = body.pipeline_mode or _LIVE_SETTINGS.pipeline_mode
    use_cap_classifier = (
        body.use_cap_classifier
        if body.use_cap_classifier is not None
        else _LIVE_SETTINGS.use_cap_classifier
    )

    if product_category is None or product_sub_type is None or container_contents is None:
        try:
            run_repo = ProductionRunRepository(motor_db)
            active_run = await run_repo.get_active_run_for_sku(body.sku)
            if active_run is not None and active_run.product_id:
                product_repo = ProductRepository(motor_db)
                product = await product_repo.get_product_by_sku(body.sku)
                if product is not None:
                    product_category = product_category or product.product_category
                    product_sub_type = product_sub_type or product.product_sub_type
                    container_contents = container_contents or product.container_contents
            else:
                log.warning(
                    "no_active_run_for_sku",
                    extra={"sku": body.sku},
                )
        except Exception as exc:  # noqa: BLE001 — never block inspection on DB errors
            log.warning(
                "sub_type_resolution_failed",
                extra={"sku": body.sku, "error": str(exc)},
            )

    if pipeline_mode == "yolo_fill_level":
        try:
            result = _get_yolo_fill_pipeline().inspect(
                frame=frame,
                product_id=body.product_id,
                sku=body.sku,
                attempt_count=body.attempt_count,
                use_cap_classifier=use_cap_classifier,
                product_category=product_category,
                product_sub_type=product_sub_type,
                container_contents=container_contents,
            )
        except Exception as exc:  # noqa: BLE001 - surface model-load/inference errors as 503s
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(exc),
            ) from exc
    else:
        result = pipeline.inspect(
            frame=frame,
            product_id=body.product_id,
            sku=body.sku,
            attempt_count=body.attempt_count,
            product_category=product_category,
            product_sub_type=product_sub_type,
            container_contents=container_contents,
        )
    repo = InspectionRepository(db)
    repo.save(result, attempt_count=body.attempt_count)
    await _publish_live_result(result)
    return result


@router.get(
    "/live-settings",
    response_model=LiveInspectionSettings,
    dependencies=[Depends(verify_api_key)],
    summary="Get live inspection pipeline settings",
)
def get_live_settings() -> LiveInspectionSettings:
    return _LIVE_SETTINGS


@router.patch(
    "/live-settings",
    response_model=LiveInspectionSettings,
    dependencies=[Depends(verify_api_key)],
    summary="Update live inspection pipeline settings",
)
def update_live_settings(body: LiveInspectionSettingsUpdate) -> LiveInspectionSettings:
    global _LIVE_SETTINGS
    data = _LIVE_SETTINGS.model_dump()
    if body.pipeline_mode is not None:
        data["pipeline_mode"] = body.pipeline_mode
    if body.use_cap_classifier is not None:
        data["use_cap_classifier"] = body.use_cap_classifier
    _LIVE_SETTINGS = LiveInspectionSettings(**data)
    return _LIVE_SETTINGS


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
