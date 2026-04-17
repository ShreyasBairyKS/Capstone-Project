"""
api/routers/runs.py — Production run management endpoints.

Endpoints (prefix: /api/v1/runs):
    POST   /runs                   — Start a new production run
    GET    /runs/active            — Current active run (any SKU) for the dashboard status bar
    GET    /runs/active/{sku}      — Active run for a specific SKU
    PATCH  /runs/{run_id}/end      — End a run (supervisor+)
"""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from motor.motor_asyncio import AsyncIOMotorDatabase

from api.dependencies import get_motor_db, require_role, verify_api_key
from database.mongo_models import ProductionRun, RunCreate
from database.repositories.product_repository import ProductRepository
from database.repositories.production_run_repository import ProductionRunRepository

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1/runs",
    tags=["Production Runs"],
    dependencies=[Depends(verify_api_key)],
)


# ---------------------------------------------------------------------------
# POST /runs — start a new production run
# ---------------------------------------------------------------------------

@router.post(
    "",
    response_model=ProductionRun,
    status_code=status.HTTP_201_CREATED,
    summary="Start a new production run",
    description=(
        "Creates and activates a production run for the specified SKU. "
        "Returns 404 if the SKU is not registered. "
        "Returns 409 if an active run already exists for that SKU."
    ),
)
async def start_run(
    data: RunCreate,
    db: Annotated[AsyncIOMotorDatabase, Depends(get_motor_db)],
) -> ProductionRun:
    # Verify the SKU is registered as a product
    product_repo = ProductRepository(db)
    product = await product_repo.get_product_by_sku(data.sku)
    if product is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No product registered with SKU '{data.sku}'. Register the product first.",
        )

    run_repo = ProductionRunRepository(db)
    try:
        return await run_repo.start_run(data, product_id=str(product.id) if product.id else None)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# GET /runs/active — global active run (dashboard status bar)
# ---------------------------------------------------------------------------

@router.get(
    "/active",
    response_model=ProductionRun | None,
    summary="Get any active production run",
    description=(
        "Returns the most recently started active production run across all SKUs, "
        "or null if no run is currently active. "
        "The dashboard status bar polls this endpoint every 10 seconds."
    ),
)
async def get_any_active_run(
    db: Annotated[AsyncIOMotorDatabase, Depends(get_motor_db)],
) -> ProductionRun | None:
    repo = ProductionRunRepository(db)
    return await repo.get_any_active_run()


# ---------------------------------------------------------------------------
# GET /runs/active/{sku} — active run for a specific SKU
# ---------------------------------------------------------------------------

@router.get(
    "/active/{sku}",
    response_model=ProductionRun | None,
    summary="Get active production run for a specific SKU",
    description=(
        "Returns the active run for the specified SKU, "
        "or null if no run is active for that SKU."
    ),
)
async def get_active_run_for_sku(
    sku: str,
    db: Annotated[AsyncIOMotorDatabase, Depends(get_motor_db)],
) -> ProductionRun | None:
    repo = ProductionRunRepository(db)
    return await repo.get_active_run_for_sku(sku)


# ---------------------------------------------------------------------------
# PATCH /runs/{run_id}/end — end a run
# ---------------------------------------------------------------------------

class _EndRunBody:
    """Inline request body for ending a run."""
    pass


from pydantic import BaseModel


class EndRunRequest(BaseModel):
    status: str = "completed"  # 'completed' or 'aborted'


@router.patch(
    "/{run_id}/end",
    response_model=ProductionRun,
    summary="End an active production run",
    description=(
        "Transitions an active run to 'completed' or 'aborted'. "
        "Returns 404 if the run_id is not found or the run is not active. "
        "Requires **supervisor** or **admin** role."
    ),
)
async def end_run(
    run_id: str,
    body: EndRunRequest,
    db: Annotated[AsyncIOMotorDatabase, Depends(get_motor_db)],
    _role: Annotated[str, Depends(require_role("supervisor"))],
) -> ProductionRun:
    repo = ProductionRunRepository(db)
    try:
        result = await repo.end_run(run_id=run_id, status=body.status)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No active run found with run_id='{run_id}'.",
        )
    return result
