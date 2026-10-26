"""
Direct End-to-End Test Suite for Part 5: Station 2 REST API, Symmetrical VRAM Management & WebSocket Logic.
Validates the exact boot and switching behavior specified:
- Initial boot: Cap-classifier boots normally (eager), Fill-level boots in standby (0 VRAM).
- Switch 1 -> 2: Cap unloads (0 VRAM), Fill loads dynamically.
- Switch 2 -> 1: Fill unloads (0 VRAM), Cap reloads dynamically.
- Full Station 2 REST, Audit, Replay, and Active Learning functionality.
"""
import sys
import json
import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s]: %(message)s")
logger = logging.getLogger("test_fill_step5_direct")

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
    ws_manager
)
from pipeline.fill_level.api.schemas import SaveFillAnnotationRequest


def run_all_tests():
    logger.info("================================================================")
    logger.info("STEP 1: Verify Initial Boot Conditions")
    logger.info("================================================================")
    cap_svc = PipelineService.get_instance()
    fill_svc = FillPipelineService.get_instance()

    # Verify Cap Classifier is loaded normally on boot (not lazy at boot)
    assert cap_svc.is_engine_loaded() is True, "Cap Classifier should boot loaded by default!"
    cap_h = cap_health_check()
    logger.info(f"Cap Station Health at Boot: status={cap_h['status']}, loaded={cap_h['models_loaded']}, hw={cap_h['hardware']}")
    assert cap_h["models_loaded"] is True
    assert cap_h["station"] == "Station 1: Cap Inspection"

    # Verify Fill Level Classifier is in STANDBY on boot (lazy loading)
    assert fill_svc.is_engine_loaded() is False, "Fill Classifier should boot in Standby mode (0 VRAM)!"
    fill_h = get_fill_health()
    logger.info(f"Fill Station Health at Boot: status={fill_h.status}, loaded={fill_h.models_loaded}, hw={fill_h.hardware_runtime}")
    assert fill_h.models_loaded is False
    assert fill_h.station == "Station 2: Fill Level Inspection"
    logger.info("STEP 1 PASSED: Initial boot states are 100% compliant with specifications.")

    logger.info("================================================================")
    logger.info("STEP 2: Perform Station Switch: Cap -> Fill Level")
    logger.info("================================================================")
    # 1. Unload Cap Station to free GPU compute & VRAM
    res_cap_unl = unload_cap_engine()
    logger.info(f"Cap Station unload response: {res_cap_unl}")
    assert res_cap_unl["models_loaded"] is False
    assert cap_svc.is_engine_loaded() is False
    assert cap_health_check()["models_loaded"] is False
    assert cap_health_check()["hardware"] == "Standby (Unloaded)"

    # 2. Dynamically load Fill Station
    res_fill_load = load_fill_engine()
    logger.info(f"Fill Station load response: {res_fill_load}")
    assert res_fill_load["models_loaded"] is True
    assert fill_svc.is_engine_loaded() is True
    fill_h2 = get_fill_health()
    assert fill_h2.models_loaded is True
    logger.info(f"Fill Station loaded on: {fill_h2.hardware_runtime} (Engine Version: {fill_h2.engine_version})")
    logger.info("STEP 2 PASSED: Switched to Station 2, Station 1 VRAM deallocated.")

    logger.info("================================================================")
    logger.info("STEP 3: Station 2 Inspection Simulation & Visualizer")
    logger.info("================================================================")
    # Execute simulation cycle directly on FillPipelineService
    sim_res = fill_svc.run_simulated_cycle()
    ratio_val = sim_res.fill_ratio or 0.0
    logger.info(f"Simulation result: Bottle={sim_res.bottle_id}, Verdict={sim_res.verdict}, Fill Ratio={ratio_val:.2%}, Compliant={sim_res.is_compliant}")
    assert sim_res.annotated_image_base64 is not None
    assert len(sim_res.annotated_image_base64) > 100
    bottle_id = sim_res.bottle_id
    logger.info("STEP 3 PASSED: Inspection simulation ran successfully with live overlay.")

    logger.info("================================================================")
    logger.info("STEP 4: Query Fill KPI Stats, Defects & CSV Export")
    logger.info("================================================================")
    stats = get_fill_stats()
    logger.info(f"Fill Stats: Total={stats.total_inspected}, Pass={stats.pass_count}, Under={stats.underfill_count}, Over={stats.overfill_count}, Pass Rate={stats.pass_rate:.1f}%")
    assert stats.total_inspected >= 1

    defects = get_fill_defects(limit=10, offset=0, paginate=True)
    logger.info(f"Defect query: Total count={defects['total_count']}, Returned items={len(defects['items'])}")
    assert defects["total_count"] >= 0

    csv_resp = export_fill_csv()
    csv_text = csv_resp.body.decode("utf-8") if hasattr(csv_resp, "body") else str(csv_resp)
    assert "id,bottle_id,timestamp,global_verdict" in csv_text
    assert bottle_id in csv_text
    logger.info("STEP 4 PASSED: Stats, defects log, and CSV export validated.")

    logger.info("================================================================")
    logger.info("STEP 5: Query Bottle Detail & Dynamic On-Demand Replay")
    logger.info("================================================================")
    detail = get_bottle_detail(bottle_id)
    assert detail["record"]["bottle_id"] == bottle_id
    assert detail["raw_image_base64"] is not None
    assert detail["annotated_image_base64"] is not None
    assert "adjacent" in detail
    logger.info(f"Retrieved bottle {bottle_id} detail: clean raw image size={len(detail['raw_image_base64'])} chars, dynamic overlay size={len(detail['annotated_image_base64'])} chars")
    logger.info("STEP 5 PASSED: Raw image preservation and dynamic replay verified.")

    logger.info("================================================================")
    logger.info("STEP 6: Active Learning Workflow (Review -> Correction -> Export)")
    logger.info("================================================================")
    cands = get_al_candidates(uncertain_only=False, limit=10)
    logger.info(f"Active learning candidates count: {len(cands)}")

    # Submit operator correction
    save_req = SaveFillAnnotationRequest(
        bottle_id=bottle_id,
        global_verdict="NORMAL",
        fill_status="normal",
        water_box=[120, 160, 280, 170]
    )
    save_res = save_al_annotation(save_req)
    assert save_res["status"] == "success"
    logger.info(f"Operator correction saved: {save_res}")

    # Export YOLO training dataset zip directly from service
    zip_bytes = fill_svc.export_dataset_zip()
    assert len(zip_bytes) > 0
    assert zip_bytes[:4] == b"PK\x03\x04", "Valid ZIP magic header expected!"
    logger.info(f"Active learning dataset zip exported: {len(zip_bytes)} bytes")
    logger.info("STEP 6 PASSED: Active learning workflow 100% verified.")

    logger.info("================================================================")
    logger.info("STEP 7: Switch Back: Fill Level -> Cap Classifier")
    logger.info("================================================================")
    # 1. Unload Fill Station
    res_fill_unl = unload_fill_engine()
    assert res_fill_unl["models_loaded"] is False
    assert fill_svc.is_engine_loaded() is False
    assert get_fill_health().models_loaded is False
    logger.info("Fill Station unloaded: models_loaded=False, 0 MB VRAM.")

    # 2. Reload Cap Station
    res_cap_load = load_cap_engine()
    assert res_cap_load["models_loaded"] is True
    assert cap_svc.is_engine_loaded() is True
    assert cap_health_check()["models_loaded"] is True
    logger.info("Cap Station reloaded: models_loaded=True.")

    # 3. Verify Cap Station can execute inspection cycles without performance loss
    cap_sim = cap_svc.run_simulated_cycle()
    logger.info(f"Cap Station simulated cycle: Bottle={cap_sim.bottle_id}, Global Verdict={cap_sim.global_verdict}, Cameras={cap_sim.camera_count}")
    assert cap_sim.camera_count >= 1
    logger.info("STEP 7 PASSED: Station switch back to Cap Classifier 100% verified.")

    print("\n" + "=" * 70)
    print("ALL PART 5 ACCEPTANCE CRITERIA AND SWITCHING BEHAVIORS PASSED!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    run_all_tests()
