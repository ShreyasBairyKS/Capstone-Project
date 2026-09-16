"""
SQLite Telemetry Database Manager for Part 5: Logging, Defect Audit & Telemetry.
Manages schema initialization, ACID transaction inserts, KPI rollups, and indexed queries.
"""
import os
import json
import time
import sqlite3
import logging
import threading
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any

from pipeline.audit.schemas import AuditConfig, BottleAuditRecord, KPISummary, InspectionFilter

logger = logging.getLogger("pipeline.audit.db")


class DatabaseManager:
    """
    Thread-safe SQLite database manager for industrial inspection telemetry.
    Configured with Write-Ahead Logging (WAL) for concurrent read/write throughput.
    """

    def __init__(self, config: Optional[AuditConfig] = None):
        self.config = config or AuditConfig()
        self.base_dir = Path(self.config.base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.base_dir / self.config.db_filename
        self._lock = threading.Lock()
        self._conn: Optional[sqlite3.Connection] = None
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Returns the thread-safe connection with WAL mode and row factory enabled."""
        if self._conn is None:
            self._conn = sqlite3.connect(
                str(self.db_path),
                timeout=20.0,
                check_same_thread=False
            )
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL;")
            self._conn.execute("PRAGMA synchronous=NORMAL;")
            self._conn.execute("PRAGMA foreign_keys=ON;")
        return self._conn

    def _init_db(self) -> None:
        """Initializes tables and indexes if they do not exist."""
        with self._lock:
            conn = self._get_connection()
            with conn:
                cursor = conn.cursor()
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS inspections (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        bottle_id TEXT UNIQUE NOT NULL,
                        timestamp TEXT NOT NULL,
                        epoch_time REAL NOT NULL,
                        global_verdict TEXT NOT NULL,
                        is_compliant INTEGER NOT NULL,
                        trigger_actuator INTEGER NOT NULL,
                        primary_defect TEXT,
                        primary_defect_confidence REAL,
                        flagged_cameras TEXT,
                        decision_reason TEXT,
                        total_latency_ms REAL,
                        camera_count INTEGER,
                        bundle_dir TEXT
                    );
                """)

                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS defect_events (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        bottle_id TEXT NOT NULL,
                        camera_id TEXT NOT NULL,
                        defect_name TEXT NOT NULL,
                        confidence REAL NOT NULL,
                        FOREIGN KEY (bottle_id) REFERENCES inspections(bottle_id) ON DELETE CASCADE
                    );
                """)

                # Fast indexes for dashboard analytics and filtering
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_inspections_bottle_id ON inspections(bottle_id);")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_inspections_verdict ON inspections(global_verdict);")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_inspections_epoch ON inspections(epoch_time);")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_inspections_defect ON inspections(primary_defect);")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_defect_events_name ON defect_events(defect_name);")
            logger.info(f"Initialized inspection telemetry database at {self.db_path}")

    def insert_record(
        self,
        record: BottleAuditRecord,
        defects: Optional[List[Tuple[str, str, float]]] = None
    ) -> int:
        """Inserts an inspection record and associated defect events."""
        with self._lock:
            conn = self._get_connection()
            with conn:
                cursor = conn.cursor()
                flagged_json = json.dumps(record.flagged_cameras)
                cursor.execute("""
                    INSERT OR REPLACE INTO inspections (
                        bottle_id, timestamp, epoch_time, global_verdict,
                        is_compliant, trigger_actuator, primary_defect,
                        primary_defect_confidence, flagged_cameras, decision_reason,
                        total_latency_ms, camera_count, bundle_dir
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    record.bottle_id,
                    record.timestamp,
                    record.epoch_time,
                    record.global_verdict,
                    1 if record.is_compliant else 0,
                    1 if record.trigger_actuator else 0,
                    record.primary_defect,
                    record.primary_defect_confidence,
                    flagged_json,
                    record.decision_reason,
                    record.total_latency_ms,
                    record.camera_count,
                    record.bundle_dir
                ))
                row_id = cursor.lastrowid

                if defects:
                    cursor.executemany("""
                        INSERT INTO defect_events (bottle_id, camera_id, defect_name, confidence)
                        VALUES (?, ?, ?, ?)
                    """, [(record.bottle_id, cam, name, conf) for cam, name, conf in defects])

                return row_id

    def get_by_bottle_id(self, bottle_id: str) -> Optional[BottleAuditRecord]:
        """Retrieves a single inspection record by unique bottle_id."""
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM inspections WHERE bottle_id = ?", (bottle_id,))
            row = cursor.fetchone()
            if not row:
                return None
            return self._row_to_record(row)

    def query_inspections(self, f: Optional[InspectionFilter] = None) -> List[BottleAuditRecord]:
        """Queries historical records with flexible filtering and pagination."""
        f = f or InspectionFilter()
        clauses = []
        params = []

        if f.verdict:
            clauses.append("global_verdict = ?")
            params.append(f.verdict.upper())
        if f.defect_type:
            clauses.append("primary_defect = ?")
            params.append(f.defect_type)
        if f.start_epoch is not None:
            clauses.append("epoch_time >= ?")
            params.append(f.start_epoch)
        if f.end_epoch is not None:
            clauses.append("epoch_time <= ?")
            params.append(f.end_epoch)

        where_str = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        query = f"""
            SELECT * FROM inspections
            {where_str}
            ORDER BY epoch_time DESC
            LIMIT ? OFFSET ?
        """
        params.extend([f.limit, f.offset])

        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()
            return [self._row_to_record(r) for r in rows]

    def get_kpis(self, time_window_seconds: Optional[float] = None) -> KPISummary:
        """Computes rolling industrial KPIs."""
        where_str = ""
        params = []
        if time_window_seconds is not None and time_window_seconds > 0:
            min_epoch = time.time() - time_window_seconds
            where_str = "WHERE epoch_time >= ?"
            params.append(min_epoch)

        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()

            agg_query = f"""
                SELECT 
                    COUNT(*) as total,
                    SUM(CASE WHEN global_verdict = 'PASS' THEN 1 ELSE 0 END) as passed,
                    SUM(CASE WHEN global_verdict = 'REJECT' THEN 1 ELSE 0 END) as rejected,
                    SUM(CASE WHEN global_verdict = 'UNCERTAIN' THEN 1 ELSE 0 END) as uncertain,
                    SUM(CASE WHEN global_verdict = 'INSPECTION_FAILED' THEN 1 ELSE 0 END) as failed,
                    AVG(total_latency_ms) as avg_latency
                FROM inspections
                {where_str}
            """
            cursor.execute(agg_query, params)
            row = cursor.fetchone()

            total = row["total"] or 0
            passed = row["passed"] or 0
            rejected = row["rejected"] or 0
            uncertain = row["uncertain"] or 0
            failed = row["failed"] or 0
            avg_lat = float(row["avg_latency"]) if row["avg_latency"] is not None else 0.0

            pass_rate = round((passed / total * 100.0), 2) if total > 0 else 100.0
            reject_rate = round((rejected / total * 100.0), 2) if total > 0 else 0.0

            defect_query = f"""
                SELECT defect_name, COUNT(*) as cnt
                FROM defect_events
                { "WHERE bottle_id IN (SELECT bottle_id FROM inspections WHERE epoch_time >= ?)" if time_window_seconds else "" }
                GROUP BY defect_name
                ORDER BY cnt DESC
            """
            cursor.execute(defect_query, params if time_window_seconds else [])
            defect_rows = cursor.fetchall()
            defect_breakdown = {r["defect_name"]: r["cnt"] for r in defect_rows}

            return KPISummary(
                total_inspected=total,
                passed=passed,
                rejected=rejected,
                uncertain=uncertain,
                failed=failed,
                pass_rate_pct=pass_rate,
                reject_rate_pct=reject_rate,
                defect_breakdown=defect_breakdown,
                average_latency_ms=round(avg_lat, 2)
            )

    def close(self) -> None:
        """Closes the database connection cleanly."""
        with self._lock:
            if self._conn is not None:
                try:
                    self._conn.close()
                except Exception:
                    pass
                self._conn = None
                logger.info("Database connection closed cleanly.")

    def _row_to_record(self, row: sqlite3.Row) -> BottleAuditRecord:
        """Helper to convert sqlite3.Row to BottleAuditRecord schema."""
        flagged = []
        try:
            flagged = json.loads(row["flagged_cameras"]) if row["flagged_cameras"] else []
        except Exception:
            pass

        return BottleAuditRecord(
            id=row["id"],
            bottle_id=row["bottle_id"],
            timestamp=row["timestamp"],
            epoch_time=row["epoch_time"],
            global_verdict=row["global_verdict"],
            is_compliant=bool(row["is_compliant"]),
            trigger_actuator=bool(row["trigger_actuator"]),
            primary_defect=row["primary_defect"],
            primary_defect_confidence=row["primary_defect_confidence"],
            flagged_cameras=flagged,
            decision_reason=row["decision_reason"],
            total_latency_ms=row["total_latency_ms"],
            camera_count=row["camera_count"],
            bundle_dir=row["bundle_dir"]
        )
