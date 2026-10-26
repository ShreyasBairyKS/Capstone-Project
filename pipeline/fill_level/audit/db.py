"""
SQLite WAL Database Handler for Fill Level Inspection Station.
Optimized for concurrent high-speed writes, indexed queries, and real-time KPI metrics.
"""
import time
import sqlite3
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any, Union

from .schemas import (
    FillKPISummary,
    FillInspectionFilter,
    FillInspectionRecord,
)

logger = logging.getLogger("pipeline.fill_level.audit.db")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class FillAuditDB:
    """
    Dedicated SQLite storage for Station 2 (Fill Level Inspection).
    Thread-safe connection lifecycle with SQLite WAL pragma tuning.
    """

    def __init__(self, db_path: Union[str, Path] = "audit/fill_level.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Creates connection with high-performance industrial pragmas."""
        conn = sqlite3.connect(str(self.db_path), timeout=5.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        conn.execute("PRAGMA cache_size = -64000;")  # 64 MB
        conn.execute("PRAGMA busy_timeout = 5000;")
        return conn

    def _init_db(self):
        """Initializes tables and indexes."""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS fill_inspections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    bottle_id TEXT UNIQUE NOT NULL,
                    timestamp TEXT NOT NULL,
                    epoch_time REAL NOT NULL,
                    global_verdict TEXT NOT NULL,
                    fill_status TEXT NOT NULL,
                    fill_ratio REAL,
                    water_y INTEGER,
                    bottle_h INTEGER,
                    water_conf REAL,
                    bottle_conf REAL,
                    is_compliant INTEGER NOT NULL,
                    trigger_actuator INTEGER NOT NULL,
                    decision_reason TEXT,
                    total_latency_ms REAL,
                    bundle_dir TEXT
                );
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_fill_epoch ON fill_inspections(epoch_time DESC);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_fill_verdict ON fill_inspections(global_verdict);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_fill_status ON fill_inspections(fill_status);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_fill_bottle ON fill_inspections(bottle_id);")
            conn.commit()
        logger.info(f"FillAuditDB initialized at: {self.db_path}")

    def insert_inspection(self, data: Dict[str, Any]) -> int:
        """Inserts inspection record and returns rowid."""
        query = """
            INSERT OR REPLACE INTO fill_inspections (
                bottle_id, timestamp, epoch_time, global_verdict, fill_status,
                fill_ratio, water_y, bottle_h, water_conf, bottle_conf,
                is_compliant, trigger_actuator, decision_reason, total_latency_ms, bundle_dir
            ) VALUES (
                :bottle_id, :timestamp, :epoch_time, :global_verdict, :fill_status,
                :fill_ratio, :water_y, :bottle_h, :water_conf, :bottle_conf,
                :is_compliant, :trigger_actuator, :decision_reason, :total_latency_ms, :bundle_dir
            );
        """
        with self._get_connection() as conn:
            cur = conn.execute(query, data)
            conn.commit()
            return cur.lastrowid

    def query_inspections(self, filt: FillInspectionFilter) -> List[FillInspectionRecord]:
        """Queries records with pagination and filtering."""
        conditions = []
        params: Dict[str, Any] = {}

        if filt.verdict:
            conditions.append("global_verdict = :verdict")
            params["verdict"] = filt.verdict.upper()

        if filt.fill_status:
            conditions.append("fill_status = :fill_status")
            params["fill_status"] = filt.fill_status.lower()

        if filt.search:
            conditions.append("bottle_id LIKE :search")
            params["search"] = f"%{filt.search}%"

        if filt.start_epoch is not None:
            conditions.append("epoch_time >= :start_epoch")
            params["start_epoch"] = filt.start_epoch

        if filt.end_epoch is not None:
            conditions.append("epoch_time <= :end_epoch")
            params["end_epoch"] = filt.end_epoch

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        query = f"""
            SELECT * FROM fill_inspections
            {where_clause}
            ORDER BY epoch_time DESC
            LIMIT :limit OFFSET :offset;
        """
        params["limit"] = filt.limit
        params["offset"] = filt.offset

        with self._get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
            return [self._row_to_record(r) for r in rows]

    def count_inspections(self, filt: FillInspectionFilter) -> int:
        """Counts matching records."""
        conditions = []
        params: Dict[str, Any] = {}

        if filt.verdict:
            conditions.append("global_verdict = :verdict")
            params["verdict"] = filt.verdict.upper()

        if filt.fill_status:
            conditions.append("fill_status = :fill_status")
            params["fill_status"] = filt.fill_status.lower()

        if filt.search:
            conditions.append("bottle_id LIKE :search")
            params["search"] = f"%{filt.search}%"

        if filt.start_epoch is not None:
            conditions.append("epoch_time >= :start_epoch")
            params["start_epoch"] = filt.start_epoch

        if filt.end_epoch is not None:
            conditions.append("epoch_time <= :end_epoch")
            params["end_epoch"] = filt.end_epoch

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        query = f"SELECT COUNT(*) FROM fill_inspections {where_clause};"

        with self._get_connection() as conn:
            return conn.execute(query, params).fetchone()[0]

    def get_kpis(self, window_seconds: Optional[float] = None) -> FillKPISummary:
        """Calculates real-time manufacturing KPIs."""
        params: Dict[str, Any] = {}
        where_clause = ""
        if window_seconds is not None:
            min_epoch = time.time() - window_seconds
            where_clause = "WHERE epoch_time >= :min_epoch"
            params["min_epoch"] = min_epoch

        query = f"""
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN global_verdict = 'PASS' THEN 1 ELSE 0 END) as pass_c,
                SUM(CASE WHEN fill_status = 'underfill' THEN 1 ELSE 0 END) as under_c,
                SUM(CASE WHEN fill_status = 'overfill' THEN 1 ELSE 0 END) as over_c,
                SUM(CASE WHEN fill_status = 'no_water' THEN 1 ELSE 0 END) as dry_c,
                SUM(CASE WHEN global_verdict = 'UNCERTAIN' THEN 1 ELSE 0 END) as unc_c,
                AVG(CASE WHEN fill_ratio IS NOT NULL THEN fill_ratio ELSE NULL END) as avg_ratio,
                AVG(total_latency_ms) as avg_latency
            FROM fill_inspections
            {where_clause};
        """
        with self._get_connection() as conn:
            row = conn.execute(query, params).fetchone()
            total = row["total"] or 0
            pass_c = row["pass_c"] or 0
            under_c = row["under_c"] or 0
            over_c = row["over_c"] or 0
            dry_c = row["dry_c"] or 0
            unc_c = row["unc_c"] or 0
            avg_ratio = row["avg_ratio"] or 0.0
            avg_latency = row["avg_latency"] or 0.0

            pass_rate = round((pass_c / total * 100.0), 2) if total > 0 else 0.0

            return FillKPISummary(
                total_inspected=total,
                pass_count=pass_c,
                underfill_count=under_c,
                overfill_count=over_c,
                dry_count=dry_c,
                uncertain_count=unc_c,
                pass_rate=pass_rate,
                mean_fill_ratio=round(float(avg_ratio), 3),
                mean_latency_ms=round(float(avg_latency), 2)
            )

    def get_ratio_histogram(self, bins: int = 10, window_seconds: Optional[float] = None) -> List[Dict[str, Any]]:
        """Counts measured fill ratios (bottles with a located meniscus) into equal-width bins over [0, 1]."""
        bins = max(2, min(bins, 50))
        params: Dict[str, Any] = {"bins": bins}
        where = "WHERE fill_ratio IS NOT NULL AND fill_status IN ('normal','underfill','overfill','borderline')"
        if window_seconds is not None:
            where += " AND epoch_time >= :min_epoch"
            params["min_epoch"] = time.time() - window_seconds
        query = f"""
            SELECT MIN(CAST(MAX(fill_ratio, 0.0) * :bins AS INTEGER), :bins - 1) AS bin_idx, COUNT(*) AS cnt
            FROM fill_inspections
            {where}
            GROUP BY bin_idx;
        """
        with self._get_connection() as conn:
            counts = {row["bin_idx"]: row["cnt"] for row in conn.execute(query, params).fetchall()}
        width = 1.0 / bins
        return [
            {"lo": round(i * width, 4), "hi": round((i + 1) * width, 4), "count": int(counts.get(i, 0))}
            for i in range(bins)
        ]

    def get_bottle(self, bottle_id: str) -> Optional[FillInspectionRecord]:
        """Fetches a specific bottle inspection record by bottle_id."""
        query = "SELECT * FROM fill_inspections WHERE bottle_id = :b_id LIMIT 1;"
        with self._get_connection() as conn:
            row = conn.execute(query, {"b_id": bottle_id}).fetchone()
            return self._row_to_record(row) if row else None

    def get_adjacent_bottles(self, bottle_id: str) -> Dict[str, Optional[str]]:
        """Resolves previous and next bottle IDs for S2 Replay stepper navigation."""
        with self._get_connection() as conn:
            cur_row = conn.execute(
                "SELECT epoch_time FROM fill_inspections WHERE bottle_id = :b_id LIMIT 1;",
                {"b_id": bottle_id}
            ).fetchone()
            if not cur_row:
                return {"prev_id": None, "next_id": None}

            cur_epoch = cur_row["epoch_time"]

            prev_row = conn.execute(
                "SELECT bottle_id FROM fill_inspections WHERE epoch_time < :ep ORDER BY epoch_time DESC LIMIT 1;",
                {"ep": cur_epoch}
            ).fetchone()

            next_row = conn.execute(
                "SELECT bottle_id FROM fill_inspections WHERE epoch_time > :ep ORDER BY epoch_time ASC LIMIT 1;",
                {"ep": cur_epoch}
            ).fetchone()

            return {
                "prev_id": prev_row["bottle_id"] if prev_row else None,
                "next_id": next_row["bottle_id"] if next_row else None
            }

    def update_bottle_annotation(
        self,
        bottle_id: str,
        global_verdict: Optional[str] = None,
        fill_status: Optional[str] = None,
        is_compliant: Optional[bool] = None
    ) -> bool:
        """Updates ground-truth classification from S3 Active Learning triage."""
        updates = []
        params: Dict[str, Any] = {"b_id": bottle_id}

        if global_verdict is not None:
            updates.append("global_verdict = :verdict")
            params["verdict"] = global_verdict.upper()

        if fill_status is not None:
            updates.append("fill_status = :status")
            params["status"] = fill_status.lower()

        if is_compliant is not None:
            updates.append("is_compliant = :compliant")
            params["compliant"] = 1 if is_compliant else 0

        if not updates:
            return False

        query = f"UPDATE fill_inspections SET {', '.join(updates)} WHERE bottle_id = :b_id;"
        with self._get_connection() as conn:
            cur = conn.execute(query, params)
            conn.commit()
            return cur.rowcount > 0

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> FillInspectionRecord:
        return FillInspectionRecord(
            id=row["id"],
            bottle_id=row["bottle_id"],
            timestamp=row["timestamp"],
            epoch_time=row["epoch_time"],
            global_verdict=row["global_verdict"],
            fill_status=row["fill_status"],
            fill_ratio=row["fill_ratio"],
            water_y=row["water_y"],
            bottle_h=row["bottle_h"],
            water_conf=row["water_conf"],
            bottle_conf=row["bottle_conf"],
            is_compliant=bool(row["is_compliant"]),
            trigger_actuator=bool(row["trigger_actuator"]),
            decision_reason=row["decision_reason"],
            total_latency_ms=row["total_latency_ms"],
            bundle_dir=row["bundle_dir"]
        )
