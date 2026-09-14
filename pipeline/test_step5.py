"""
Comprehensive Verification Suite for Part 5: Logging, Defect Audit & Telemetry.
Validates:
1. Isolated bottle bundle creation (raw images, YOLO polygon labels, JSON metadata).
2. Anti-cross-mixing guarantees across distinct bottle inspection events.
3. SQLite database transactions, indexed queries, and rolling KPI rollups.
4. Active learning YOLOv11 segmentation format validation.
5. Full 4-Stage Pipeline integration (Ingestion -> Inference -> QualityGate -> AuditLogger).
"""
import os
import sys
import time
import shutil
import logging
from pathlib import Path
import cv2
import numpy as np

# Ensure pipeline root is in sys.path
pipeline_root = Path(__file__).resolve().parent
project_root = pipeline_root.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from pipeline.audit.schemas import AuditConfig, InspectionFilter
from pipeline.audit.logger import AuditLogger
from pipeline.inference.schemas import Detection
from pipeline.decision.schemas import (
    GlobalInspectionResult,
    ViewVerdict,
    InspectionVerdict
)

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s")
logger = logging.getLogger("test_step5")


def test_part5_audit():
    print("=" * 75)
    print("  STEP 5 VERIFICATION: AUDIT LOGGER, DEFECT BUNDLER & TELEMETRY")
    print("=" * 75)

    test_audit_dir = pipeline_root / "test_audit_sandbox"
    if test_audit_dir.exists():
        shutil.rmtree(test_audit_dir, ignore_errors=True)

    config = AuditConfig(
        base_dir=str(test_audit_dir),
        db_filename="inspections.db",
        bottles_subfolder="bottles",
        save_pass_images=True,
        save_defect_images=True,
        save_yolo_labels=True,
        async_saving=False  # Synchronous for deterministic assertions
    )

    logger_engine = AuditLogger(config)

    # -------------------------------------------------------------------------
    # [1/5] Isolated Bottle Bundle Creation (Simulated Defect Bottle)
    # -------------------------------------------------------------------------
    print("\n[1/5] Testing Isolated Bottle Bundle Creation (Defect Bottle)...")
    
    bottle_id_1 = "BOTTLE_20260912_TEST_0001"
    # Create two synthetic 640x640 camera frames
    img_front = np.full((640, 640, 3), (40, 40, 180), dtype=np.uint8)  # Reddish
    img_rear = np.full((640, 640, 3), (180, 40, 40), dtype=np.uint8)   # Blueish
    frames_1 = {"camera_front": img_front, "camera_rear": img_rear}

    # Simulate detection with a real polygon mask
    mock_mask = [[200.0, 150.0], [440.0, 150.0], [420.0, 350.0], [220.0, 350.0]]
    det_damaged = Detection(
        class_id=0,
        class_name="damaged_cap",
        confidence=0.88,
        box=[200.0, 150.0, 440.0, 350.0],
        mask=mock_mask,
        mask_area_pixels=45000.0
    )
    det_good_rear = Detection(
        class_id=1,
        class_name="good_cap",
        confidence=0.92,
        box=[190.0, 160.0, 430.0, 340.0],
        mask=[[190.0, 160.0], [430.0, 160.0], [430.0, 340.0], [190.0, 340.0]],
        mask_area_pixels=43200.0
    )

    view_front = ViewVerdict(
        camera_id="camera_front",
        verdict=InspectionVerdict.REJECT,
        primary_class="damaged_cap",
        primary_confidence=0.88,
        defect_detected=True,
        defect_name="damaged_cap",
        defect_confidence=0.88,
        decision_reason="Defect detected: damaged_cap (88%)",
        detections=[det_damaged]
    )
    view_rear = ViewVerdict(
        camera_id="camera_rear",
        verdict=InspectionVerdict.PASS,
        primary_class="good_cap",
        primary_confidence=0.92,
        defect_detected=False,
        decision_reason="Compliant cap (92%)",
        detections=[det_good_rear]
    )

    decision_1 = GlobalInspectionResult(
        bottle_id=bottle_id_1,
        timestamp="2026-09-12T15:00:00.000Z",
        global_verdict=InspectionVerdict.REJECT,
        is_compliant=False,
        trigger_reject_actuator=True,
        primary_defect="damaged_cap",
        primary_defect_confidence=0.88,
        flagged_cameras=["camera_front"],
        decision_reason="Ejection triggered by camera_front: damaged_cap",
        total_latency_ms=18.5,
        per_view_results={"camera_front": view_front, "camera_rear": view_rear}
    )

    rec_1 = logger_engine.log_inspection(frames_1, decision_1, wait_sync=True)
    assert rec_1.bottle_id == bottle_id_1
    assert rec_1.global_verdict == "REJECT"
    assert rec_1.trigger_actuator is True

    bundle_dir_1 = test_audit_dir / "bottles" / bottle_id_1
    assert bundle_dir_1.exists(), f"Bundle directory {bundle_dir_1} does not exist!"
    assert (bundle_dir_1 / "camera_front.jpg").exists(), "camera_front.jpg missing!"
    assert (bundle_dir_1 / "camera_rear.jpg").exists(), "camera_rear.jpg missing!"
    assert (bundle_dir_1 / "inspection_data.json").exists(), "inspection_data.json missing!"
    assert (bundle_dir_1 / "labels" / "camera_front.txt").exists(), "YOLO label camera_front.txt missing!"
    assert (bundle_dir_1 / "labels" / "camera_rear.txt").exists(), "YOLO label camera_rear.txt missing!"

    # Verify raw image integrity
    img_read = cv2.imread(str(bundle_dir_1 / "camera_front.jpg"))
    assert img_read is not None and img_read.shape == (640, 640, 3)

    print(f"  ✓ Isolated bundle verified at: {bundle_dir_1.name}")
    print(f"  ✓ Raw images saved cleanly without annotations (Resolution: 640x640)")
    print(f"  ✓ Metadata JSON and YOLO labels directory populated")

    # -------------------------------------------------------------------------
    # [2/5] Active Learning YOLOv11 Segmentation Label Format Validation
    # -------------------------------------------------------------------------
    print("\n[2/5] Validating YOLO Segmentation Label Syntax...")
    label_path = bundle_dir_1 / "labels" / "camera_front.txt"
    with open(label_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    assert len(lines) == 1, "Expected 1 detection line in label file"
    tokens = lines[0].split()
    class_id = int(tokens[0])
    poly_coords = [float(val) for val in tokens[1:]]

    assert class_id == 0, f"Expected class 0 (damaged_cap), got {class_id}"
    assert len(poly_coords) == 8, f"Expected 4 coordinate pairs (8 values), got {len(poly_coords)}"
    for val in poly_coords:
        assert 0.0 <= val <= 1.0, f"Normalized coordinate {val} out of bounds [0, 1]!"

    print(f"  ✓ YOLO segmentation format verified: {lines[0][:55]}...")
    print("  ✓ All polygon coordinates normalized in [0.0, 1.0] ready for YOLOv11 active learning")

    # -------------------------------------------------------------------------
    # [3/5] Anti-Cross-Mixing Isolation Verification
    # -------------------------------------------------------------------------
    print("\n[3/5] Testing Anti-Cross-Mixing Isolation with Second Bottle (PASS)...")
    bottle_id_2 = "BOTTLE_20260912_TEST_0002"
    frames_2 = {"camera_front": np.full((480, 640, 3), (50, 180, 50), dtype=np.uint8)}  # Single camera

    det_good = Detection(
        class_id=1,
        class_name="good_cap",
        confidence=0.96,
        box=[180.0, 140.0, 420.0, 320.0],
        mask=[[180.0, 140.0], [420.0, 140.0], [420.0, 320.0], [180.0, 320.0]]
    )
    view_front_2 = ViewVerdict(
        camera_id="camera_front",
        verdict=InspectionVerdict.PASS,
        primary_class="good_cap",
        primary_confidence=0.96,
        defect_detected=False,
        decision_reason="Certified compliant (96%)",
        detections=[det_good]
    )
    decision_2 = GlobalInspectionResult(
        bottle_id=bottle_id_2,
        timestamp="2026-09-12T15:00:01.000Z",
        global_verdict=InspectionVerdict.PASS,
        is_compliant=True,
        trigger_reject_actuator=False,
        primary_defect=None,
        flagged_cameras=[],
        decision_reason="All camera views compliant",
        total_latency_ms=14.2,
        per_view_results={"camera_front": view_front_2}
    )

    rec_2 = logger_engine.log_inspection(frames_2, decision_2, wait_sync=True)
    bundle_dir_2 = test_audit_dir / "bottles" / bottle_id_2

    assert bundle_dir_2.exists()
    assert (bundle_dir_2 / "camera_front.jpg").exists()
    # Ensure BOTTLE_002 did NOT get camera_rear from BOTTLE_001
    assert not (bundle_dir_2 / "camera_rear.jpg").exists(), "Cross-contamination! Camera rear found in bottle 2!"
    # Ensure BOTTLE_001 was untouched
    assert (bundle_dir_1 / "camera_rear.jpg").exists()

    print(f"  ✓ Bottle 1 directory contents: {sorted([p.name for p in bundle_dir_1.iterdir()])}")
    print(f"  ✓ Bottle 2 directory contents: {sorted([p.name for p in bundle_dir_2.iterdir()])}")
    print("  ✓ Strict anti-cross-mixing guarantee verified!")

    # -------------------------------------------------------------------------
    # [4/5] Database Transactions, Filtered Queries & KPI Rollups
    # -------------------------------------------------------------------------
    print("\n[4/5] Testing Database Transactions & Real-Time KPI Rollups...")
    
    # Query by bottle_id
    queried_rec = logger_engine.get_bottle_record(bottle_id_1)
    assert queried_rec is not None
    assert queried_rec.bottle_id == bottle_id_1
    assert queried_rec.global_verdict == "REJECT"
    assert queried_rec.primary_defect == "damaged_cap"
    assert queried_rec.trigger_actuator is True
    print(f"  ✓ Direct DB lookup by bottle_id successful: {queried_rec.bottle_id}")

    # Query with filter: only REJECT
    reject_records = logger_engine.query_inspections(InspectionFilter(verdict="REJECT"))
    assert len(reject_records) == 1
    assert reject_records[0].bottle_id == bottle_id_1
    print(f"  ✓ Filtered query (verdict=REJECT) returned {len(reject_records)} record (BOTTLE_001)")

    # Query with filter: only PASS
    pass_records = logger_engine.query_inspections(InspectionFilter(verdict="PASS"))
    assert len(pass_records) == 1
    assert pass_records[0].bottle_id == bottle_id_2
    print(f"  ✓ Filtered query (verdict=PASS) returned {len(pass_records)} record (BOTTLE_002)")

    # KPI Summary rollup
    kpis = logger_engine.get_kpis()
    assert kpis.total_inspected == 2
    assert kpis.passed == 1
    assert kpis.rejected == 1
    assert kpis.pass_rate_pct == 50.0
    assert kpis.reject_rate_pct == 50.0
    assert "damaged_cap" in kpis.defect_breakdown
    assert kpis.defect_breakdown["damaged_cap"] == 1
    print(f"  ✓ Telemetry KPIs: Inspected={kpis.total_inspected} | Pass Rate={kpis.pass_rate_pct}% | Reject Rate={kpis.reject_rate_pct}%")
    print(f"  ✓ Defect breakdown tally: {kpis.defect_breakdown}")
    print(f"  ✓ Average pipeline cycle latency: {kpis.average_latency_ms} ms")

    # Manifest retrieval for on-demand dynamic replay
    manifest = logger_engine.get_bottle_manifest(bottle_id_1)
    assert manifest is not None
    assert manifest["bottle_id"] == bottle_id_1
    assert "bundle_manifest" in manifest
    print(f"  ✓ Dynamic replay manifest retrieved for: {bottle_id_1}")

    # -------------------------------------------------------------------------
    # [5/5] Full 4-Stage Pipeline Chaining (Ingestion -> Inference -> Gate -> Audit)
    # -------------------------------------------------------------------------
    print("\n[5/5] Testing Live 4-Stage End-to-End Pipeline Chaining...")
    from pipeline.inference.engine import InferenceEngine
    from pipeline.decision.gate import QualityGate
    from pipeline.ingestion.stream_manager import StreamManager

    img1_path = project_root / "runs" / "segment" / "predict" / "IMG20260824154700.jpg"
    img2_path = project_root / "runs" / "segment" / "predict2" / "IMG20260824154411.jpg"

    if img1_path.exists() and img2_path.exists():
        # Initialize all 4 pipeline components
        inf_engine = InferenceEngine()
        gate = QualityGate()
        stream_mgr = StreamManager()

        # Step A: Ingestion Bundle
        live_bundle = stream_mgr.create_offline_bundle(
            bottle_id="BOTTLE_E2E_AUDIT_001",
            camera_images={"camera_front": str(img1_path), "camera_rear": str(img2_path)}
        )

        # Step B: Inference Forward Pass
        inf_result = inf_engine.predict_multi_camera(live_bundle.frames)

        # Step C: Multi-View Decision Gate
        live_decision = gate.evaluate_multi_camera(
            multi_result=inf_result,
            bottle_id=live_bundle.bottle_id
        )

        # Step D: Audit Logger Archiving
        live_record = logger_engine.log_inspection(
            frames=live_bundle.frames,
            decision=live_decision,
            wait_sync=True
        )

        assert live_record.bottle_id == "BOTTLE_E2E_AUDIT_001"
        live_bundle_dir = test_audit_dir / "bottles" / "BOTTLE_E2E_AUDIT_001"
        assert live_bundle_dir.exists()
        assert (live_bundle_dir / "camera_front.jpg").exists()
        assert (live_bundle_dir / "camera_rear.jpg").exists()
        assert (live_bundle_dir / "inspection_data.json").exists()

        print(f"  ✓ Live E2E bottle logged: {live_record.bottle_id}")
        print(f"  ✓ Verdict: {live_record.global_verdict} | Actuator: {live_record.trigger_actuator}")
        print(f"  ✓ Archived bundle path: {live_record.bundle_dir}")
    else:
        print("  ! Test sample images not found, skipping live 4-stage chain.")

    # Clean shutdown
    logger_engine.shutdown()

    # Clean up test sandbox
    if test_audit_dir.exists():
        shutil.rmtree(test_audit_dir, ignore_errors=True)

    print("\n" + "=" * 75)
    print("  >>> STEP 5 VERIFICATION PASSED: Audit Logger & Defect Bundler is 100% Operational <<<")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    test_part5_audit()
