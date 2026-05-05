"""
api/routers/products.py — Product registration and profile management endpoints.

Endpoints (prefix: /api/v1/products):
    POST   /products              — Register a new product (supervisor+)
    GET    /products              — Paginated list of products
    GET    /products/sku-profiles — Available YAML profile names for the frontend dropdown
    GET    /products/{sku}        — Single product by SKU
    PATCH  /products/{sku}        — Update product with optimistic concurrency (supervisor+)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from motor.motor_asyncio import AsyncIOMotorDatabase

from api.dependencies import get_motor_db, require_role, verify_api_key
from core.config import settings
from database.mongo_models import Product, ProductCreate, ProductUpdate
from database.repositories.product_repository import ProductRepository

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1/products",
    tags=["Products"],
    dependencies=[Depends(verify_api_key)],
)

# Allowed enum values — stub until Collaborator A delivers core/schemas.py enums
_VALID_CATEGORIES = {"beverage", "food", "general"}
_VALID_SUB_TYPES = {"transparent_bottle", "rigid_can", "flexible_wrapper", "rigid_box"}
_VALID_CONTENTS = {"liquid", "solid"}

# SKU regex: lowercase alphanumeric + underscores, 3–64 chars
import re
_SKU_RE = re.compile(r"^[a-z0-9_]{3,64}$")


def _validate_product_create(data: ProductCreate) -> None:
    """Raise HTTP 422 if the payload violates domain rules."""
    if not _SKU_RE.match(data.sku):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="sku must be 3–64 lowercase alphanumeric characters or underscores.",
        )
    if data.product_category not in _VALID_CATEGORIES:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"product_category must be one of {sorted(_VALID_CATEGORIES)}.",
        )
    if data.product_sub_type not in _VALID_SUB_TYPES:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"product_sub_type must be one of {sorted(_VALID_SUB_TYPES)}.",
        )
    if data.container_contents not in _VALID_CONTENTS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"container_contents must be one of {sorted(_VALID_CONTENTS)}.",
        )
    # Validate sku_profile_name exists on disk
    profile_dir: Path = settings.SKU_PROFILES_DIR
    profile_path = profile_dir / f"{data.sku_profile_name}.yaml"
    if not profile_path.exists():
        existing = [p.stem for p in profile_dir.glob("*.yaml")]
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"sku_profile_name '{data.sku_profile_name}' does not match any YAML file "
                f"in {profile_dir}. Available: {sorted(existing)}."
            ),
        )
    # Strip whitespace on string fields
    data.name = data.name.strip()
    if data.description:
        data.description = data.description.strip()
    if data.qr_code:
        data.qr_code = data.qr_code.strip()


# ---------------------------------------------------------------------------
# POST /products — register a new product
# ---------------------------------------------------------------------------

@router.post(
    "",
    response_model=Product,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new product",
    description=(
        "Creates a product document in MongoDB. "
        "Validates SKU format, category/sub-type/contents enums, and the existence of "
        "the referenced SKU profile YAML. Requires **supervisor** or **admin** role."
    ),
)
async def create_product(
    data: ProductCreate,
    db: Annotated[AsyncIOMotorDatabase, Depends(get_motor_db)],
    _role: Annotated[str, Depends(require_role("supervisor"))],
) -> Product:
    _validate_product_create(data)
    repo = ProductRepository(db)
    try:
        return await repo.create_product(data)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# GET /products/sku-profiles — list available YAML profile names
# ---------------------------------------------------------------------------

@router.get(
    "/sku-profiles",
    response_model=list[str],
    summary="List available SKU profile names",
    description=(
        "Returns the filename stems of all YAML files present in the SKU profiles directory. "
        "Used by the frontend ProductRegistration form to populate the SKU Profile dropdown."
    ),
)
async def list_sku_profiles() -> list[str]:
    profile_dir: Path = settings.SKU_PROFILES_DIR
    if not profile_dir.exists():
        logger.warning("SKU profiles directory not found: %s", profile_dir)
        return []
    return sorted(p.stem for p in profile_dir.glob("*.yaml"))


# ---------------------------------------------------------------------------
# GET /products — paginated list
# ---------------------------------------------------------------------------

@router.get(
    "",
    response_model=list[Product],
    summary="List all products",
    description="Returns a paginated list of registered products, optionally filtered by category.",
)
async def list_products(
    db: Annotated[AsyncIOMotorDatabase, Depends(get_motor_db)],
    skip: int = Query(default=0, ge=0, description="Number of documents to skip"),
    limit: int = Query(default=50, ge=1, le=200, description="Maximum documents to return"),
    category: str | None = Query(default=None, description="Filter by product_category"),
) -> list[Product]:
    repo = ProductRepository(db)
    return await repo.list_products(skip=skip, limit=limit, category=category)


# ---------------------------------------------------------------------------
# GET /products/{sku} — single product
# ---------------------------------------------------------------------------

@router.get(
    "/{sku}",
    response_model=Product,
    summary="Get product by SKU",
    description="Returns the product document for the given SKU identifier, or 404 if not found.",
)
async def get_product(
    sku: str,
    db: Annotated[AsyncIOMotorDatabase, Depends(get_motor_db)],
) -> Product:
    repo = ProductRepository(db)
    product = await repo.get_product_by_sku(sku)
    if product is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No product found with SKU '{sku}'.",
        )
    return product


# ---------------------------------------------------------------------------
# PATCH /products/{sku} — partial update with optimistic concurrency
# ---------------------------------------------------------------------------

@router.patch(
    "/{sku}",
    response_model=Product,
    summary="Update a product (optimistic concurrency)",
    description=(
        "Partially updates a product document. The request body must include the current "
        "`version` (__v) value. If __v has changed since the caller last read the document, "
        "the server returns 409 Conflict. Requires **supervisor** or **admin** role."
    ),
)
async def update_product(
    sku: str,
    data: ProductUpdate,
    db: Annotated[AsyncIOMotorDatabase, Depends(get_motor_db)],
    _role: Annotated[str, Depends(require_role("supervisor"))],
) -> Product:
    repo = ProductRepository(db)
    try:
        return await repo.update_product(sku, data)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
