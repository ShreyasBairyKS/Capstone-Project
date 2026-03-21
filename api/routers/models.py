"""
api/routers/models.py — Model version management endpoints.

Routes:
  GET    /models                — List all registered model versions
  POST   /models                — Register a new model version
  PATCH  /models/{id}/activate  — Promote a model version to active
  POST   /models/{id}/rollback  — Roll back to a previous version
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from api.dependencies import get_db, verify_api_key
from database.models import ModelVersion

router = APIRouter(prefix="/models", tags=["Model Versions"])


# ------------------------------------------------------------------ #
# Schemas
# ------------------------------------------------------------------ #

class ModelVersionCreate(BaseModel):
    name: str = Field(..., max_length=128)
    version_tag: str = Field(..., max_length=32)
    detector_path: str = Field(..., max_length=256)
    classifier_path: str = Field(..., max_length=256)
    map50: Optional[float] = None
    top1_accuracy: Optional[float] = None
    trained_at: Optional[datetime] = None


class ModelVersionResponse(BaseModel):
    id: int
    name: str
    version_tag: str
    detector_path: str
    classifier_path: str
    map50: Optional[float]
    top1_accuracy: Optional[float]
    is_active: bool
    trained_at: Optional[datetime]
    deployed_at: Optional[datetime]

    model_config = {"from_attributes": True}


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

def _row_to_dict(mv: ModelVersion) -> dict:
    return {
        "id": mv.id,
        "name": mv.name,
        "version_tag": mv.version_tag,
        "detector_path": mv.detector_path,
        "classifier_path": mv.classifier_path,
        "map50": mv.map50,
        "top1_accuracy": mv.top1_accuracy,
        "is_active": mv.is_active,
        "trained_at": mv.trained_at.isoformat() if mv.trained_at else None,
        "deployed_at": mv.deployed_at.isoformat() if mv.deployed_at else None,
    }


# ------------------------------------------------------------------ #
# Endpoints
# ------------------------------------------------------------------ #

@router.get(
    "",
    response_model=list[dict],
    dependencies=[Depends(verify_api_key)],
    summary="List all model versions",
)
def list_models(db: Session = Depends(get_db)) -> list[dict]:
    rows = db.execute(
        select(ModelVersion).order_by(ModelVersion.id.desc())
    ).scalars().all()
    return [_row_to_dict(r) for r in rows]


@router.post(
    "",
    response_model=dict,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(verify_api_key)],
    summary="Register a new model version",
)
def register_model(
    body: ModelVersionCreate,
    db: Session = Depends(get_db),
) -> dict:
    mv = ModelVersion(
        name=body.name,
        version_tag=body.version_tag,
        detector_path=body.detector_path,
        classifier_path=body.classifier_path,
        map50=body.map50,
        top1_accuracy=body.top1_accuracy,
        trained_at=body.trained_at,
        is_active=False,
    )
    db.add(mv)
    db.commit()
    db.refresh(mv)
    return _row_to_dict(mv)


@router.patch(
    "/{model_id}/activate",
    response_model=dict,
    dependencies=[Depends(verify_api_key)],
    summary="Promote a model version to active",
)
def activate_model(
    model_id: int,
    db: Session = Depends(get_db),
) -> dict:
    """
    Set the specified model as active and deactivate all others.
    Note: hot-swapping the in-process ONNX session requires a restart;
    this endpoint updates the database record only.
    """
    target = db.get(ModelVersion, model_id)
    if target is None:
        raise HTTPException(status_code=404, detail="Model version not found.")

    # Deactivate all others
    all_rows = db.execute(
        select(ModelVersion).where(ModelVersion.is_active == True)  # noqa: E712
    ).scalars().all()
    for row in all_rows:
        row.is_active = False

    target.is_active = True
    target.deployed_at = datetime.utcnow()
    db.commit()
    db.refresh(target)
    return _row_to_dict(target)


@router.post(
    "/{model_id}/rollback",
    response_model=dict,
    dependencies=[Depends(verify_api_key)],
    summary="Roll back: deactivate current active, activate given version",
)
def rollback_model(
    model_id: int,
    db: Session = Depends(get_db),
) -> dict:
    target = db.get(ModelVersion, model_id)
    if target is None:
        raise HTTPException(status_code=404, detail="Model version not found.")

    # Deactivate all currently active rows
    active_rows = db.execute(
        select(ModelVersion).where(ModelVersion.is_active == True)  # noqa: E712
    ).scalars().all()
    for row in active_rows:
        row.is_active = False

    # Activate rollback target
    target.is_active = True
    target.deployed_at = datetime.utcnow()
    db.commit()
    db.refresh(target)
    return {**_row_to_dict(target), "message": f"Rolled back to version {target.version_tag}"}
