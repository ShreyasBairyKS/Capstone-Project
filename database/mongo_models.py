"""
database/mongo_models.py — Motor (async MongoDB) document models for VisionFood QAI.

These models live alongside the existing SQLAlchemy layer (database/models.py) and are
used exclusively for Product and ProductionRun documents stored in MongoDB.

Design decisions:
- PyObjectId wraps bson.ObjectId so Pydantic can serialise it to str.
- ConfigDict(populate_by_name=True) allows both `id` and `_id` aliases to work.
- All timestamps are UTC-aware datetime objects.
- __v (version counter) enables optimistic concurrency for PATCH operations.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Annotated, Any

from bson import ObjectId
from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# PyObjectId — serialise MongoDB _id as plain str in JSON responses
# ---------------------------------------------------------------------------

class PyObjectId(str):
    """Pydantic-compatible wrapper around bson.ObjectId."""

    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v: Any) -> str:
        if isinstance(v, ObjectId):
            return str(v)
        if isinstance(v, str) and ObjectId.is_valid(v):
            return v
        raise ValueError(f"Invalid ObjectId: {v!r}")

    @classmethod
    def __get_pydantic_core_schema__(cls, _source_type: Any, _handler: Any):
        from pydantic_core import core_schema
        return core_schema.no_info_plain_validator_function(cls.validate)


# ---------------------------------------------------------------------------
# Shared date field model (used in Product.expected_dates)
# ---------------------------------------------------------------------------

class ExpectedDateField(BaseModel):
    """One OCR date field the pipeline should locate and verify."""
    name: str = Field(..., description="e.g. 'expiry_date', 'mfg_date', 'best_before'")
    format: str = Field(..., description="e.g. 'MM/YYYY', 'DDMMMYYYY', 'DD/MM/YYYY'")
    value: str | None = Field(
        default=None,
        description="Expected printed value (optional — used for strict equality checks)",
    )


# ---------------------------------------------------------------------------
# Product document
# ---------------------------------------------------------------------------

class Product(BaseModel):
    """
    MongoDB document stored in the `products` collection.
    Maps to the /api/v1/products endpoints.
    """
    model_config = ConfigDict(
        populate_by_name=True,
        arbitrary_types_allowed=True,
        json_encoders={ObjectId: str},
    )

    id: PyObjectId | None = Field(default=None, alias="_id")

    # Core identity
    sku: str = Field(..., description="Unique SKU identifier, e.g. 'bottle_250ml'")
    name: str = Field(..., description="Human-readable product name")
    description: str | None = Field(default=None)

    # Classification (used for pipeline routing)
    product_category: str = Field(
        ...,
        description="Top-level category: 'beverage', 'food', or 'general'",
    )
    product_sub_type: str = Field(
        ...,
        description="Container sub-type: 'transparent_bottle', 'rigid_can', 'flexible_wrapper', 'rigid_box'",
    )
    container_contents: str = Field(
        ...,
        description="Physical state of contents: 'liquid' or 'solid'",
    )

    # YAML profile linkage (loaded by Collaborator A's profile manager)
    sku_profile_name: str = Field(
        ...,
        description="Filename stem of the SKU YAML profile (e.g. 'bottle_250ml')",
    )

    # Verification references — supplied to BarcodeVerifier and LabelOCRVerifier
    qr_code: str | None = Field(
        default=None,
        description="Expected QR code value for barcode verification",
    )
    expected_dates: list[ExpectedDateField] = Field(
        default_factory=list,
        description="Expected OCR date fields for label verification",
    )

    # Metadata
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    version: int = Field(
        default=0,
        alias="__v",
        serialization_alias="__v",
        description="Optimistic concurrency version counter",
    )


class ProductCreate(BaseModel):
    """Request body for POST /api/v1/products."""
    sku: str
    name: str
    description: str | None = None
    product_category: str
    product_sub_type: str
    container_contents: str
    sku_profile_name: str
    qr_code: str | None = None
    expected_dates: list[ExpectedDateField] = Field(default_factory=list)


class ProductUpdate(BaseModel):
    """Request body for PATCH /api/v1/products/{sku} — all fields optional."""
    name: str | None = None
    description: str | None = None
    product_category: str | None = None
    product_sub_type: str | None = None
    container_contents: str | None = None
    sku_profile_name: str | None = None
    qr_code: str | None = None
    expected_dates: list[ExpectedDateField] | None = None
    # Caller must echo the current __v; server rejects if stale
    version: int = Field(..., description="Current __v value for optimistic concurrency check")


# ---------------------------------------------------------------------------
# ProductionRun document
# ---------------------------------------------------------------------------

class ProductionRun(BaseModel):
    """
    MongoDB document stored in the `production_runs` collection.
    Maps to the /api/v1/runs endpoints.
    """
    model_config = ConfigDict(
        populate_by_name=True,
        arbitrary_types_allowed=True,
        json_encoders={ObjectId: str},
    )

    id: PyObjectId | None = Field(default=None, alias="_id")

    # Unique run identifier (UUID4 string — safe for URL path params)
    run_id: str = Field(default_factory=lambda: str(uuid.uuid4()))

    # Product linkage
    sku: str = Field(..., description="SKU of the product being inspected in this run")
    product_id: str | None = Field(
        default=None,
        description="MongoDB _id of the associated Product document",
    )

    # Timing
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    ended_at: datetime | None = Field(default=None)

    # Status lifecycle: active → completed | aborted
    status: str = Field(default="active", description="'active', 'completed', or 'aborted'")

    # Operator info
    operator_id: str | None = Field(
        default=None,
        description="Username or identifier of the operator who started the run",
    )

    # Running counters (incremented by the inspection pipeline per frame)
    inspection_count: int = Field(default=0)
    defect_count: int = Field(default=0)


class RunCreate(BaseModel):
    """Request body for POST /api/v1/runs."""
    sku: str
    operator_id: str | None = None


# ---------------------------------------------------------------------------
# MongoDB index creation — call once at application startup
# ---------------------------------------------------------------------------

async def create_product_run_indexes(db) -> None:
    """
    Create required indexes on the `products` and `production_runs` collections.

    Args:
        db: AsyncIOMotorDatabase instance (injected via get_motor_db).
    """
    # products: unique SKU lookup
    await db["products"].create_index("sku", unique=True)
    await db["products"].create_index("product_category")
    await db["products"].create_index("product_sub_type")

    # production_runs: active-run queries and run_id URL lookups
    await db["production_runs"].create_index("run_id", unique=True)
    await db["production_runs"].create_index([("sku", 1), ("status", 1)])
    await db["production_runs"].create_index("status")
    await db["production_runs"].create_index("started_at")
