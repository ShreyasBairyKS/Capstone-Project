"""
Audit Logger Coordinator for Part 5: Logging, Defect Audit & Telemetry.
Combines BottleBundler, DatabaseManager, asynchronous background saving,
and high-level telemetry query APIs.
"""
import time
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from concurrent.futures import ThreadPoolExecutor, Future
import numpy as np

from pipeline.audit.schemas import AuditConfig, BottleAuditRecord, KPISummary, InspectionFilter
from pipeline.audit.bundler import BottleBundler
from pipeline.audit.db import DatabaseManager
from pipeline.decision.schemas import GlobalInspectionResult, InspectionVerdict

logger = logging.getLogger("pipeline.audit.logger")


class AuditLogger:
    """
    Central industrial audit subsystem.
    Persists unannotated raw frames in bottle-isolated directories,
    auto-generates YOLO active learning polygon labels,
    logs ACID transactions into SQLite, and provides real-time KPI rollups.
    """

    def __init__(self, config: Optional[AuditConfig] = None):
        self.config = config or AuditConfig()
        self.db = DatabaseManager(self.config)
        self.bundler = BottleBundler(self.config)
        self._executor = (
            ThreadPoolExecutor(
                max_workers=self.config.thread_pool_workers,
                thread_name_prefix="AuditWorker"
            )
            if self.config.async_saving else None
        )
        logger.info(f"AuditLogger initialized (base_dir={self.config.base_dir}, async={self.config.async_saving})")

    def log_inspection(
        self,
        frames: Dict[str, np.ndarray],
        decision: GlobalInspectionResult,
        wait_sync: bool = False
    ) -> BottleAuditRecord:
        """
        Logs a completed multi-view inspection event.

        Args:
            frames: Multi-camera frames mapping (camera_id -> raw numpy image).
            decision: Authoritative GlobalInspectionResult from Part 2 Quality Gate.
            wait_sync: If True, blocks until disk bundling finishes even if async_saving is enabled.

        Returns:
            Structured BottleAuditRecord.
        """
        epoch_now = time.time()

        # Extract all detected defect instances for fine-grained defect indexing
        defect_events: List[Tuple[str, str, float]] = []
        for cam_id, view in decision.per_view_results.items():
            if view.defect_detected and view.defect_name:
                defect_events.append((cam_id, view.defect_name, view.defect_confidence or 0.0))

        # Determine bundle path
        safe_id = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in decision.bottle_id)
        bundle_rel_dir = str(Path(self.config.base_dir) / self.config.bottles_subfolder / safe_id).replace("\\", "/")

        record = BottleAuditRecord(
            bottle_id=decision.bottle_id,
            timestamp=decision.timestamp,
            epoch_time=epoch_now,
            global_verdict=decision.global_verdict.value if hasattr(decision.global_verdict, "value") else str(decision.global_verdict),
            is_compliant=decision.is_compliant,
            trigger_actuator=decision.trigger_reject_actuator,
            primary_defect=decision.primary_defect,
            primary_defect_confidence=decision.primary_defect_confidence,
            flagged_cameras=decision.flagged_cameras,
            decision_reason=decision.decision_reason,
            total_latency_ms=decision.total_latency_ms,
            camera_count=decision.camera_count,
            bundle_dir=bundle_rel_dir
        )

        def _do_bundle_and_db():
            try:
                # 1. Archive unannotated raw frames, YOLO labels, and JSON
                self.bundler.bundle_bottle(
                    bottle_id=decision.bottle_id,
                    frames=frames,
                    decision=decision
                )
                # 2. Insert into SQLite
                row_id = self.db.insert_record(record, defect_events)
                record.id = row_id
            except Exception as e:
                logger.error(f"Failed to bundle inspection {decision.bottle_id}: {e}", exc_info=True)

        if self._executor and not wait_sync:
            # Non-blocking async execution
            self._executor.submit(_do_bundle_and_db)
        else:
            # Synchronous execution
            _do_bundle_and_db()

        return record

    def get_kpis(self, time_window_seconds: Optional[float] = None) -> KPISummary:
        """Retrieves real-time manufacturing KPIs."""
        return self.db.get_kpis(time_window_seconds)

    def query_inspections(self, f: Optional[InspectionFilter] = None) -> List[BottleAuditRecord]:
        """Queries historical records with filtering and pagination."""
        return self.db.query_inspections(f)

    def get_bottle_record(self, bottle_id: str) -> Optional[BottleAuditRecord]:
        """Retrieves DB metadata for a specific bottle."""
        return self.db.get_by_bottle_id(bottle_id)

    def get_bottle_manifest(self, bottle_id: str) -> Optional[Dict[str, Any]]:
        """
        Reads inspection_data.json from the isolated bottle bundle folder.
        Used by Part 4 (Visualizer) and Web Dashboard for on-demand dynamic replay.
        """
        safe_id = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in bottle_id)
        json_path = Path(self.config.base_dir) / self.config.bottles_subfolder / safe_id / "inspection_data.json"
        if not json_path.exists():
            return None
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def shutdown(self, wait: bool = True) -> None:
        """Flushes background workers cleanly."""
        if self._executor:
            self._executor.shutdown(wait=wait)
            logger.info("AuditLogger background executor terminated.")
        self.db.close()
