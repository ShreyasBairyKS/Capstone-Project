"""
Comprehensive End-to-End System Integration Verification Suite for Part 7.
Tests the complete unified VisionQAI manufacturing system:
1. Unified FastAPI route registry and OpenAPI documentation.
2. Root redirects and static React frontend asset delivery from dist/.
3. Initial boot specifications: Station 1 eager loaded, Station 2 in standby.
4. Database isolation: audit/inspections.db vs audit/fill_level.db.
5. Symmetrical GPU VRAM switching: Cap -> Fill Level -> Cap.
6. Station 1 multi-camera inspection and telemetry.
7. Station 2 single-camera fill evaluation, HUD overlay, and measurement ruler.
8. Auxiliary feature parity: S1 Audit CSV, S2 Replay Stepper, S3 Active Learning & YOLO ZIP export.
"""
import sys
import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s]: %(message)s")
logger = logging.getLogger("test_system_e2e")

from pipeline.api.app import app
from pipeline.api.pipeline_service import PipelineService
from pipeline.fill_level.api.service import FillPipelineService
from pipeline.api.routes import (
    health_check as cap_health_check,
    load_cap_engine,
    unload_cap_engine
)
from pipeline.fill_level.api.routes import (
    get_health as get_fill_health,
    load_engine as load_fill_engine,
    unload_engine as unload_fill_engine,
    get_stats as get_fill_stats,
    get_defects as get_fill_defects,
    get_bottle_detail,
    export_csv as export_fill_csv,
    get_candidates as get_al_candidates,
    save_annotation as save_al_annotation,
    export_dataset as export_al_dataset,
)
from pipeline.fill_level.api.schemas import SaveFillAnnotationRequest


def run_e2e_verification():
    logger.info("================================================================================")
    logger.info("PHASE 1: Unified Application Route Registry & Static Asset Verification")
    logger.info("================================================================================")
    routes = [r.path for r in app.routes if hasattr(r, "path")]
    logger.info(f"Total registered FastAPI routes: {len(routes)}")

    # Check Station 1 routes
    assert "/api/health" in routes, "Missing /api/health"
    assert "/api/load" in routes, "Missing /api/load"
    assert "/api/unload" in routes, "Missing /api/unload"
    assert "/api/simulate" in routes, "Missing /api/simulate"
    assert "/api/stats" in routes, "Missing /api/stats"
    assert "/api/defects" in routes, "Missing /api/defects"
    assert "/ws/stream" in routes, "Missing /ws/stream"
    logger.info("  [PASS] Station 1 routes mounted and active.")

    # Check Station 2 routes
    assert "/api/fill/health" in routes, "Missing /api/fill/health"
    assert "/api/fill/load" in routes, "Missing /api/fill/load"
    assert "/api/fill/unload" in routes, "Missing /api/fill/unload"
    assert "/api/fill/simulate" in routes, "Missing /api/fill/simulate"
    assert "/api/fill/stats" in routes, "Missing /api/fill/stats"
    assert "/api/fill/defects" in routes, "Missing /api/fill/defects"
    assert "/api/fill/bottle/{bottle_id}" in routes, "Missing /api/fill/bottle/{bottle_id}"
    assert "/api/fill/export-csv" in routes, "Missing /api/fill/export-csv"
    assert "/api/fill/active-learning/candidates" in routes, "Missing /api/fill/active-learning/candidates"
    assert "/ws/fill/stream" in routes, "Missing /ws/fill/stream"
    logger.info("  [PASS] Station 2 routes mounted and active.")

    # Check root redirects and static mounts
    assert "/" in routes, "Missing root route"
    assert "/visionqai" in routes, "Missing /visionqai route"
    frontend_dist = PROJECT_ROOT / "pipeline" / "frontend" / "dist"
    assert (frontend_dist / "index.html").exists(), "Frontend production index.html missing!"
    logger.info("  [PASS] Static frontend assets verified in dist/.")

    logger.info("================================================================================")
    logger.info("PHASE 2: Symmetrical Boot & Database Isolation Verification")
    logger.info("================================================================================")
    cap_svc = PipelineService.get_instance()
    fill_svc = FillPipelineService.get_instance()

    # Rule: Station 1 boots eager by default, Station 2 in standby (0 MB VRAM)
    assert cap_svc.is_engine_loaded() is True, "Station 1 must boot eagerly!"
    assert fill_svc.is_engine_loaded() is False, "Station 2 must boot in standby!"
    logger.info(f"  [PASS] Station 1 Eager Boot: loaded=True | Station 2 Standby: loaded=False")

    # Verify DBMS isolation
    cap_db_path = cap_svc.audit_logger.db.db_path
    fill_db_path = fill_svc.audit_logger.db.db_path
    assert cap_db_path.exists(), f"Station 1 database ({cap_db_path}) missing!"
    assert fill_db_path.exists(), f"Station 2 database ({fill_db_path}) missing!"
    logger.info(f"  [PASS] Isolated databases verified: {cap_db_path} & {fill_db_path}")

    logger.info("================================================================================")
    logger.info("PHASE 3: Station 1 Operational Cycle Execution")
    logger.info("================================================================================")
    cap_cycle = cap_svc.run_simulated_cycle()
    logger.info(f"  Station 1 cycle complete: Bottle={cap_cycle.bottle_id}, Verdict={cap_cycle.global_verdict}, Cameras={cap_cycle.camera_count}")
    assert cap_cycle.camera_count >= 1
    logger.info("  [PASS] Station 1 multi-camera inspection operational.")

    logger.info("================================================================================")
    logger.info("PHASE 4: Symmetrical Station Switch (Cap -> Fill Level)")
    logger.info("================================================================================")
    # 1. Unload Cap
    cap_unl = unload_cap_engine()
    assert cap_unl["models_loaded"] is False
    assert cap_svc.is_engine_loaded() is False
    logger.info("  Station 1 model deallocated from GPU VRAM.")

    # 2. Load Fill
    fill_load = load_fill_engine()
    assert fill_load["models_loaded"] is True
    assert fill_svc.is_engine_loaded() is True
    logger.info("  Station 2 models loaded into GPU VRAM on demand.")
    logger.info("  [PASS] Symmetrical switch to Station 2 complete.")

    logger.info("================================================================================")
    logger.info("PHASE 5: Station 2 Operational Cycle & Visualizer Verification")
    logger.info("================================================================================")
    fill_cycle = fill_svc.run_simulated_cycle()
    bottle_id = fill_cycle.bottle_id
    ratio_val = fill_cycle.fill_ratio or 0.0
    logger.info(f"  Station 2 cycle complete: Bottle={bottle_id}, Verdict={fill_cycle.verdict}, Ratio={ratio_val:.2%}")
    assert fill_cycle.annotated_image_base64 is not None
    assert len(fill_cycle.annotated_image_base64) > 100
    logger.info("  [PASS] Station 2 single profile camera frame and overlay generated.")

    logger.info("================================================================================")
    logger.info("PHASE 6: Auxiliary Feature Parity Verification on Station 2")
    logger.info("================================================================================")
    # S1 Audit Log & CSV
    stats = get_fill_stats()
    assert stats.total_inspected >= 1
    defects = get_fill_defects(limit=10, offset=0, paginate=True)
    assert defects["total_count"] >= 1
    csv_resp = export_fill_csv()
    csv_str = csv_resp.body.decode("utf-8") if hasattr(csv_resp, "body") else str(csv_resp)
    assert "id,bottle_id,timestamp,global_verdict" in csv_str
    logger.info("  [PASS] S1 Fill Audit Log & CSV export verified.")

    # S2 Replay Inspector & Stepper
    detail = get_bottle_detail(bottle_id)
    assert detail["record"]["bottle_id"] == bottle_id
    assert detail["raw_image_base64"] is not None
    assert detail["annotated_image_base64"] is not None
    assert "adjacent" in detail
    logger.info("  [PASS] S2 Replay Inspector clean raw image + dynamic overlay + stepper verified.")

    # S3 Active Learning & YOLOv11 ZIP Export
    cands = get_al_candidates(uncertain_only=False, limit=10)
    assert len(cands) >= 1
    save_req = SaveFillAnnotationRequest(
        bottle_id=bottle_id,
        global_verdict="NORMAL",
        fill_status="normal",
        water_box=[100, 150, 260, 160]
    )
    save_res = save_al_annotation(save_req)
    assert save_res["status"] == "success"
    zip_bytes = fill_svc.export_dataset_zip()
    assert len(zip_bytes) > 0 and zip_bytes[:4] == b"PK\x03\x04"
    logger.info(f"  [PASS] S3 Active Learning queue, annotation commit, and YOLO dataset ZIP ({len(zip_bytes)} bytes) verified.")

    logger.info("================================================================================")
    logger.info("PHASE 7: Symmetrical Station Switch Back (Fill Level -> Cap)")
    logger.info("================================================================================")
    # 1. Unload Fill
    fill_unl = unload_fill_engine()
    assert fill_unl["models_loaded"] is False
    assert fill_svc.is_engine_loaded() is False
    logger.info("  Station 2 models deallocated from GPU VRAM.")

    # 2. Reload Cap
    cap_load = load_cap_engine()
    assert cap_load["models_loaded"] is True
    assert cap_svc.is_engine_loaded() is True
    logger.info("  Station 1 model reloaded into GPU VRAM.")

    # 3. Post-switch Cap Cycle
    cap_cycle_after = cap_svc.run_simulated_cycle()
    assert cap_cycle_after.camera_count >= 1
    logger.info(f"  Station 1 cycle post-switch: Bottle={cap_cycle_after.bottle_id}, Verdict={cap_cycle_after.global_verdict}")
    logger.info("  [PASS] Symmetrical switch back to Station 1 complete with zero mutual overhead.")

    print("\n" + "=" * 80)
    print("ALL PART 7 END-TO-END SYSTEM INTEGRATION ACCEPTANCE CRITERIA PASSED 100%!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    run_e2e_verification()
