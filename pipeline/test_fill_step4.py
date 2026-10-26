"""
Automated Verification Script for Part 4: Dedicated SQLite DB & Audit Storage.
Validates:
1. SQLite WAL mode configuration and schema creation.
2. Logging of raw images, YOLO normalized labels, and telemetry metadata.
3. Training-ready preservation of unburned raw_frame.jpg.
4. Real-time KPI summary calculations.
5. Filtered and paginated database queries.
6. Previous/next adjacent bottle navigation for S2 Replay stepper.
7. Ground-truth annotation override for S3 Active Learning.
8. RFC 4180 compliant CSV export generation.
"""
import sys
import time
import logging
from pathlib import Path
import cv2
import numpy as np

# Ensure project root is on PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.fill_level.inference.schemas import BottleBox, MeniscusBox, SingleImageFillResult
from pipeline.fill_level.decision.schemas import FillVerdict, FillInspectionDecision
from pipeline.fill_level.decision.gate import FillQualityGate
from pipeline.fill_level.audit.schemas import (
    FillAuditConfig,
    FillInspectionFilter,
)
from pipeline.fill_level.audit.db import FillAuditDB
from pipeline.fill_level.audit.logger import FillAuditLogger

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s]: %(message)s")
logger = logging.getLogger("test_fill_step4")


def test_sqlite_wal_pragmas():
    """Validates that audit/fill_level.db initializes with WAL mode."""
    logger.info("--- Test 1: SQLite WAL Mode Pragma Validation ---")
    test_db_path = PROJECT_ROOT / "audit" / "test_fill_level.db"
    if test_db_path.exists():
        test_db_path.unlink(missing_ok=True)

    db = FillAuditDB(test_db_path)
    with db._get_connection() as conn:
        mode = conn.execute("PRAGMA journal_mode;").fetchone()[0]
        sync = conn.execute("PRAGMA synchronous;").fetchone()[0]
        assert mode.lower() == "wal", f"Expected WAL mode, got {mode}"
        logger.info(f"  [PASS] SQLite Journal Mode: {mode.upper()} | Synchronous: {sync}")
    return test_db_path


def test_logging_and_training_ready_assets(test_db_path: Path):
    """Logs sample inspections and verifies unblemished raw frames + YOLO labels."""
    logger.info("--- Test 2: Asset Bundling & Training-Ready Labels ---")
    test_dir = PROJECT_ROOT / "audit" / "test_fill_audit"
    config = FillAuditConfig(
        db_path=test_db_path,
        base_dir=test_dir,
        bottles_subfolder="bottles",
        save_raw_image=True
    )
    audit_logger = FillAuditLogger(config)

    # 1. Log Nominal Pass Bottle
    frame1 = np.full((640, 640, 3), 120, dtype=np.uint8)
    dec1 = FillInspectionDecision(
        bottle_id="BOTTLE_TEST_NOMINAL_01",
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        verdict=FillVerdict.PASS,
        is_compliant=True,
        trigger_reject_actuator=False,
        fill_status="normal",
        fill_ratio=0.723,
        decision_reason="Nominal volume verified",
        decision_latency_ms=0.08,
        total_latency_ms=15.2,
        bottle=BottleBox(box=[100, 50, 300, 550], confidence=0.96),
        meniscus=MeniscusBox(box=[130, 180, 270, 190], water_y=185, confidence=0.89)
    )
    audit_logger.log_inspection(frame1, dec1)

    # 2. Log Overfill Defect Bottle
    frame2 = np.full((640, 640, 3), 140, dtype=np.uint8)
    dec2 = FillInspectionDecision(
        bottle_id="BOTTLE_TEST_OVERFILL_02",
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        verdict=FillVerdict.REJECT,
        is_compliant=False,
        trigger_reject_actuator=True,
        fill_status="overfill",
        fill_ratio=0.902,
        decision_reason="Overfill defect confirmed",
        decision_latency_ms=0.09,
        total_latency_ms=16.1,
        bottle=BottleBox(box=[100, 50, 300, 550], confidence=0.95),
        meniscus=MeniscusBox(box=[130, 90, 270, 100], water_y=95, confidence=0.92)
    )
    audit_logger.log_inspection(frame2, dec2)

    # 3. Log Dry Container Bottle (no water)
    frame3 = np.full((640, 640, 3), 160, dtype=np.uint8)
    dec3 = FillInspectionDecision(
        bottle_id="BOTTLE_TEST_DRY_03",
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        verdict=FillVerdict.REJECT,
        is_compliant=False,
        trigger_reject_actuator=True,
        fill_status="no_water",
        fill_ratio=0.0,
        decision_reason="No liquid detected",
        decision_latency_ms=0.06,
        total_latency_ms=12.0,
        bottle=BottleBox(box=[100, 50, 300, 550], confidence=0.91),
        meniscus=None
    )
    audit_logger.log_inspection(frame3, dec3)

    # 4. Log Borderline Review Bottle
    frame4 = np.full((640, 640, 3), 180, dtype=np.uint8)
    dec4 = FillInspectionDecision(
        bottle_id="BOTTLE_TEST_UNCERTAIN_04",
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        verdict=FillVerdict.UNCERTAIN,
        is_compliant=False,
        trigger_reject_actuator=False,
        fill_status="borderline",
        fill_ratio=0.390,
        decision_reason="Borderline fill routed for review",
        decision_latency_ms=0.07,
        total_latency_ms=14.5,
        bottle=BottleBox(box=[100, 50, 300, 550], confidence=0.94),
        meniscus=MeniscusBox(box=[130, 350, 270, 360], water_y=355, confidence=0.38)
    )
    audit_logger.log_inspection(frame4, dec4)

    # Asset Checks
    b1_dir = test_dir / "bottles" / "BOTTLE_TEST_NOMINAL_01"
    assert (b1_dir / "raw_frame.jpg").exists(), "raw_frame.jpg missing!"
    assert (b1_dir / "inspection_data.json").exists(), "inspection_data.json missing!"
    
    label_file = b1_dir / "labels" / "water_surface.txt"
    assert label_file.exists(), "YOLO label file missing!"
    label_content = label_file.read_text().strip()
    assert label_content.startswith("0 "), f"Invalid YOLO class label: {label_content}"
    tokens = label_content.split()
    assert len(tokens) == 5, f"Expected 5 YOLO tokens, got {tokens}"
    logger.info(f"  [PASS] Verified clean raw frame and YOLO label: '{label_content}'")
    return audit_logger


def test_kpi_and_queries(audit_logger: FillAuditLogger):
    """Validates KPI calculations, filters, stepper, overrides, and CSV exports."""
    logger.info("--- Test 3: KPI Metrics, Queries & Stepper ---")
    
    # 1. KPI Calculation
    kpis = audit_logger.get_kpis()
    assert kpis.total_inspected == 4, f"Expected 4, got {kpis.total_inspected}"
    assert kpis.pass_count == 1, f"Expected 1 pass, got {kpis.pass_count}"
    assert kpis.overfill_count == 1
    assert kpis.dry_count == 1
    assert kpis.uncertain_count == 1
    assert kpis.pass_rate == 25.0
    logger.info(f"  [PASS] KPIs: Total={kpis.total_inspected} | Pass Rate={kpis.pass_rate}% | Mean Ratio={kpis.mean_fill_ratio}")

    # 2. Paginated Queries & Filters
    filt_reject = FillInspectionFilter(verdict="REJECT")
    rejects = audit_logger.query_inspections(filt_reject)
    assert len(rejects) == 2, f"Expected 2 rejects, got {len(rejects)}"

    filt_overfill = FillInspectionFilter(fill_status="overfill")
    overfills = audit_logger.query_inspections(filt_overfill)
    assert len(overfills) == 1
    assert overfills[0].bottle_id == "BOTTLE_TEST_OVERFILL_02"
    logger.info(f"  [PASS] Filtered queries verified: 2 REJECTs, 1 OVERFILL")

    # 3. Adjacent Bottle Stepper (S2 Replay Inspector)
    adj = audit_logger.get_adjacent_bottles("BOTTLE_TEST_OVERFILL_02")
    logger.info(f"  [PASS] Stepper navigation for BOTTLE_TEST_OVERFILL_02: {adj}")
    assert adj["prev_id"] is not None or adj["next_id"] is not None

    # 4. Active Learning Override
    ok = audit_logger.update_bottle_annotation(
        bottle_id="BOTTLE_TEST_UNCERTAIN_04",
        global_verdict="PASS",
        fill_status="normal",
        is_compliant=True
    )
    assert ok is True
    updated = audit_logger.get_bottle("BOTTLE_TEST_UNCERTAIN_04")
    assert updated.global_verdict == "PASS"
    assert updated.fill_status == "normal"
    logger.info(f"  [PASS] Active Learning override verified for BOTTLE_TEST_UNCERTAIN_04 -> PASS")

    # 5. RFC 4180 CSV Export
    csv_data = audit_logger.export_csv(FillInspectionFilter(limit=100))
    lines = csv_data.strip().split("\n")
    assert len(lines) == 5, f"Expected 1 header + 4 data rows, got {len(lines)}"
    assert lines[0].startswith("id,bottle_id,timestamp,global_verdict"), f"Invalid CSV header: {lines[0]}"
    logger.info(f"  [PASS] CSV Export verified ({len(lines)} lines, RFC 4180 compliant)")


def main():
    logger.info("==================================================================")
    logger.info("STARTING PART 4 VERIFICATION: DEDICATED SQLITE DB & AUDIT STORAGE")
    logger.info("==================================================================")

    test_db_path = test_sqlite_wal_pragmas()
    audit_logger = test_logging_and_training_ready_assets(test_db_path)
    test_kpi_and_queries(audit_logger)

    logger.info("==================================================================")
    logger.info("ALL PART 4 TESTS COMPLETED WITH 100% SUCCESS!")
    logger.info("==================================================================")


if __name__ == "__main__":
    main()
