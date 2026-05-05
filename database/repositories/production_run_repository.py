"""
database/repositories/production_run_repository.py — Motor async repository for ProductionRun documents.

Lifecycle:
    start_run()          → creates a new active run (blocks if another run is already active for the SKU)
    get_active_run_for_sku() / get_any_active_run() → read the current run state
    increment_counts()   → atomically bumps inspection_count / defect_count per frame
    end_run()            → transitions status to 'completed' or 'aborted'
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from database.mongo_models import ProductionRun, RunCreate

logger = logging.getLogger(__name__)


class ProductionRunRepository:
    """
    Async CRUD repository for the `production_runs` MongoDB collection.

    Usage:
        repo = ProductionRunRepository(db)
        run = await repo.start_run(RunCreate(sku="bottle_250ml"), product_id="...")
    """

    COLLECTION = "production_runs"

    def __init__(self, db) -> None:
        """
        Args:
            db: AsyncIOMotorDatabase instance (from get_motor_db dependency).
        """
        self._col = db[self.COLLECTION]

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    async def start_run(self, data: RunCreate, product_id: str | None = None) -> ProductionRun:
        """
        Create and persist a new active production run.

        Raises:
            ValueError: If an active run for the given SKU already exists (→ HTTP 409).
        """
        existing = await self.get_active_run_for_sku(data.sku)
        if existing is not None:
            raise ValueError(
                f"An active production run already exists for SKU '{data.sku}' "
                f"(run_id={existing.run_id})."
            )

        run = ProductionRun(
            sku=data.sku,
            product_id=product_id,
            operator_id=data.operator_id,
        )
        raw = run.model_dump(by_alias=True, exclude={"id"})
        result = await self._col.insert_one(raw)
        raw["_id"] = str(result.inserted_id)
        logger.info("Production run started: run_id=%s sku=%s", run.run_id, data.sku)
        return ProductionRun(**raw)

    async def end_run(
        self,
        run_id: str,
        status: str = "completed",
    ) -> ProductionRun | None:
        """
        Transition a run to 'completed' or 'aborted'.

        Args:
            run_id: UUID string of the run to end.
            status: Must be 'completed' or 'aborted'.

        Returns:
            Updated ProductionRun, or None if not found.

        Raises:
            ValueError: If status is not a valid terminal value.
        """
        if status not in ("completed", "aborted"):
            raise ValueError(f"Invalid terminal status '{status}'. Must be 'completed' or 'aborted'.")

        now = datetime.now(timezone.utc)
        result = await self._col.find_one_and_update(
            {"run_id": run_id, "status": "active"},
            {"$set": {"status": status, "ended_at": now}},
            return_document=True,
        )
        if result is None:
            logger.warning("end_run: no active run found with run_id='%s'", run_id)
            return None

        result["_id"] = str(result["_id"])
        logger.info("Production run ended: run_id=%s status=%s", run_id, status)
        return ProductionRun(**result)

    async def increment_counts(
        self,
        run_id: str,
        *,
        inspections: int = 1,
        defects: int = 0,
    ) -> None:
        """
        Atomically increment inspection and/or defect counters for a run.

        Called by the inspection pipeline once per frame result. Uses MongoDB
        $inc so concurrent increments from parallel workers are safe.

        Args:
            run_id:      UUID string of the active run.
            inspections: Number of inspections to add (default 1).
            defects:     Number of confirmed defects to add (default 0).
        """
        inc: dict = {"inspection_count": inspections}
        if defects:
            inc["defect_count"] = defects

        await self._col.update_one(
            {"run_id": run_id, "status": "active"},
            {"$inc": inc},
        )

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    async def get_active_run_for_sku(self, sku: str) -> ProductionRun | None:
        """Return the active run for a specific SKU, or None."""
        doc = await self._col.find_one({"sku": sku, "status": "active"})
        if doc is None:
            return None
        doc["_id"] = str(doc["_id"])
        return ProductionRun(**doc)

    async def get_any_active_run(self) -> ProductionRun | None:
        """
        Return any currently active run (regardless of SKU).
        Used by the dashboard status bar to show the global active run.
        """
        doc = await self._col.find_one({"status": "active"}, sort=[("started_at", -1)])
        if doc is None:
            return None
        doc["_id"] = str(doc["_id"])
        return ProductionRun(**doc)

    async def get_run_by_id(self, run_id: str) -> ProductionRun | None:
        """Return a run by its UUID run_id string, or None."""
        doc = await self._col.find_one({"run_id": run_id})
        if doc is None:
            return None
        doc["_id"] = str(doc["_id"])
        return ProductionRun(**doc)
