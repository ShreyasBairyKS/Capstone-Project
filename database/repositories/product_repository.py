"""
database/repositories/product_repository.py — Motor async repository for Product documents.

Key integration points for Collaborator A:
  - get_expected_qr(sku)    → str | None   (injected into BarcodeVerifier)
  - get_expected_dates(sku) → list[dict]   (injected into LabelOCRVerifier)

Optimistic concurrency:
  update_product() checks __v before writing. If __v has changed since the
  caller read the document, a ValueError is raised and the API layer returns 409.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from database.mongo_models import Product, ProductCreate, ProductUpdate

logger = logging.getLogger(__name__)


class ProductRepository:
    """
    Async CRUD repository for the `products` MongoDB collection.

    Usage:
        repo = ProductRepository(db)
        product = await repo.create_product(ProductCreate(...))
    """

    COLLECTION = "products"

    def __init__(self, db) -> None:
        """
        Args:
            db: AsyncIOMotorDatabase instance (from get_motor_db dependency).
        """
        self._col = db[self.COLLECTION]

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    async def create_product(self, data: ProductCreate) -> Product:
        """
        Insert a new product document.

        Raises:
            ValueError: If a product with the same SKU already exists (duplicate key).
        """
        doc = Product(
            sku=data.sku,
            name=data.name,
            description=data.description,
            product_category=data.product_category,
            product_sub_type=data.product_sub_type,
            container_contents=data.container_contents,
            sku_profile_name=data.sku_profile_name,
            qr_code=data.qr_code,
            expected_dates=data.expected_dates,
        )
        raw = doc.model_dump(by_alias=True, exclude={"id"})
        try:
            result = await self._col.insert_one(raw)
        except Exception as exc:
            if "duplicate key" in str(exc).lower() or "E11000" in str(exc):
                raise ValueError(f"Product with SKU '{data.sku}' already exists.") from exc
            raise
        raw["_id"] = str(result.inserted_id)
        return Product(**raw)

    async def update_product(self, sku: str, data: ProductUpdate) -> Product:
        """
        Patch a product document using optimistic concurrency.

        Args:
            sku:  SKU to update.
            data: ProductUpdate payload. ``data.version`` must match current ``__v``.

        Raises:
            KeyError:   If product not found.
            ValueError: If ``__v`` has changed (stale version — 409 in API layer).
        """
        existing = await self._col.find_one({"sku": sku})
        if existing is None:
            raise KeyError(f"Product '{sku}' not found.")

        if existing.get("__v", 0) != data.version:
            raise ValueError(
                f"Stale version for SKU '{sku}': "
                f"expected __v={data.version}, got __v={existing.get('__v', 0)}."
            )

        updates: dict[str, Any] = {
            "updated_at": datetime.now(timezone.utc),
            "__v": existing.get("__v", 0) + 1,
        }
        for field in (
            "name", "description", "product_category", "product_sub_type",
            "container_contents", "sku_profile_name", "qr_code", "expected_dates",
        ):
            value = getattr(data, field, None)
            if value is not None:
                if field == "expected_dates":
                    updates[field] = [ed.model_dump() for ed in value]
                else:
                    updates[field] = value

        await self._col.update_one({"sku": sku}, {"$set": updates})
        updated = await self._col.find_one({"sku": sku})
        updated["_id"] = str(updated["_id"])
        return Product(**updated)

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    async def get_product_by_sku(self, sku: str) -> Product | None:
        """Return the Product document for the given SKU, or None."""
        doc = await self._col.find_one({"sku": sku})
        if doc is None:
            return None
        doc["_id"] = str(doc["_id"])
        return Product(**doc)

    async def list_products(
        self,
        skip: int = 0,
        limit: int = 50,
        category: str | None = None,
    ) -> list[Product]:
        """
        Return a paginated list of products.

        Args:
            skip:     Number of documents to skip.
            limit:    Maximum number of documents to return.
            category: Optional filter by ``product_category``.
        """
        query: dict[str, Any] = {}
        if category:
            query["product_category"] = category

        cursor = self._col.find(query).skip(skip).limit(limit).sort("sku", 1)
        products = []
        async for doc in cursor:
            doc["_id"] = str(doc["_id"])
            products.append(Product(**doc))
        return products

    async def list_sku_profile_names(self) -> list[str]:
        """
        Return all distinct ``sku_profile_name`` values stored in the collection.
        Used by GET /api/v1/products/sku-profiles to populate the frontend dropdown.
        """
        return await self._col.distinct("sku_profile_name")

    # ------------------------------------------------------------------
    # Collaborator A integration — verifier callables
    # ------------------------------------------------------------------

    async def get_expected_qr(self, sku: str) -> str | None:
        """
        Return the expected QR code value for a SKU.

        Inject this coroutine into BarcodeVerifier so it can retrieve the
        reference value from MongoDB at inference time without importing the
        full repository.

        Example (Collaborator A usage):
            verifier = BarcodeVerifier(
                get_expected_value=lambda sku: repo.get_expected_qr(sku)
            )
        """
        doc = await self._col.find_one({"sku": sku}, {"qr_code": 1, "_id": 0})
        if doc is None:
            logger.warning("get_expected_qr: no product found for SKU '%s'", sku)
            return None
        return doc.get("qr_code")

    async def get_expected_dates(self, sku: str) -> list[dict]:
        """
        Return the expected OCR date fields for a SKU as plain dicts.

        Each dict has keys: ``name``, ``format``, ``value`` (value may be None).

        Inject this coroutine into LabelOCRVerifier so it can retrieve the
        expected dates from MongoDB at inference time.

        Example (Collaborator A usage):
            verifier = LabelOCRVerifier(
                get_expected_dates=lambda sku: repo.get_expected_dates(sku)
            )
        """
        doc = await self._col.find_one({"sku": sku}, {"expected_dates": 1, "_id": 0})
        if doc is None:
            logger.warning("get_expected_dates: no product found for SKU '%s'", sku)
            return []
        return doc.get("expected_dates", [])
