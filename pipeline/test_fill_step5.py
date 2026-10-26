"""
Test Suite for Part 5: Station 2 (Fill Level) REST API, Live WebSocket & Symmetrical VRAM Management.
Uses FastAPI TestClient to verify all REST endpoints, WebSocket streaming, and dynamic engine swapping.
"""
import os
import sys
import json
import logging
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient
from pipeline.api.app import app
from pipeline.fill_level.api.service import FillPipelineService
from pipeline.api.pipeline_service import PipelineService

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s]: %(message)s")
logger = logging.getLogger("test_fill_step5")


def run_tests():
    logger.info("Initializing FastAPI TestClient...")
    client = TestClient(app)

    logger.info("=== TEST 1: Fill Health Check on Standby ===")
    res = client.get("/api/fill/health")
    assert res.status_code == 200, f"Health check failed: {res.text}"
    health_data = res.json()
    logger.info(f"Fill health payload: {health_data}")
    assert health_data["status"] == "online"
    assert health_data["station"] == "Station 2: Fill Level Inspection"
    assert "models_loaded" in health_data
    logger.info("TEST 1 PASSED: Health check valid.")

    logger.info("=== TEST 2: Dynamic Lazy Load of Fill Engine ===")
    res = client.post("/api/fill/load")
    assert res.status_code == 200, f"Fill load failed: {res.text}"
    load_data = res.json()
    logger.info(f"Fill load payload: {load_data}")
    assert load_data["status"] == "loaded"
    assert load_data["models_loaded"] is True

    # Re-verify health reports models_loaded: true
    res = client.get("/api/fill/health")
    assert res.json()["models_loaded"] is True
    logger.info("TEST 2 PASSED: Fill engine successfully loaded on demand.")

    logger.info("=== TEST 3: Station 2 Simulate Inspection Cycle ===")
    sim_res = client.post("/api/fill/simulate")
    assert sim_res.status_code == 200, f"Simulate failed: {sim_res.text}"
    sim_data = sim_res.json()
    bottle_id = sim_data["data"]["bottle_id"]
    verdict = sim_data["data"]["verdict"]
    fill_ratio = sim_data["data"]["inference_result"]["fill_ratio"]
    logger.info(f"Simulated bottle {bottle_id}: verdict={verdict}, fill_ratio={fill_ratio:.2%}")
    assert "data" in sim_data
    assert "stats" in sim_data
    assert sim_data["data"]["annotated_view_base64"] is not None
    logger.info("TEST 3 PASSED: Simulated cycle completed with annotated frame & telemetry.")

    logger.info("=== TEST 4: Query Fill KPI Stats ===")
    stats_res = client.get("/api/fill/stats")
    assert stats_res.status_code == 200
    stats = stats_res.json()
    logger.info(f"Fill stats: {stats}")
    assert stats["total_inspections"] >= 1
    logger.info("TEST 4 PASSED: Fill stats query valid.")

    logger.info("=== TEST 5: Query Bottle Detail and Dynamic Overlay ===")
    b_res = client.get(f"/api/fill/bottle/{bottle_id}")
    assert b_res.status_code == 200
    b_data = b_res.json()
    assert b_data["bottle_id"] == bottle_id
    assert b_data["raw_image_base64"] is not None

    overlay_res = client.get(f"/api/fill/bottle/{bottle_id}/overlay")
    assert overlay_res.status_code == 200
    overlay_data = overlay_res.json()
    assert overlay_data["bottle_id"] == bottle_id
    assert overlay_data["overlay_base64"] is not None
    logger.info("TEST 5 PASSED: Bottle detail and dynamic overlay verified.")

    logger.info("=== TEST 6: Query Fill Defects and CSV Export ===")
    defects_res = client.get("/api/fill/defects")
    assert defects_res.status_code == 200
    defects_data = defects_res.json()
    assert "items" in defects_data
    assert "total" in defects_data

    csv_res = client.get("/api/fill/export-csv")
    assert csv_res.status_code == 200
    assert "text/csv" in csv_res.headers.get("content-type", "")
    assert "bottle_id" in csv_res.text
    logger.info("TEST 6 PASSED: Defects log and CSV export verified.")

    logger.info("=== TEST 7: Active Learning Endpoints ===")
    al_cands = client.get("/api/fill/active-learning/candidates")
    assert al_cands.status_code == 200
    logger.info(f"AL candidates found: {len(al_cands.json())}")

    al_detail = client.get(f"/api/fill/active-learning/bottle/{bottle_id}")
    assert al_detail.status_code == 200
    al_data = al_detail.json()
    assert al_data["bottle_id"] == bottle_id

    # Test saving annotation
    save_payload = {
        "verified_label": "NORMAL",
        "labels": al_data["labels"],
        "operator_id": "TEST_OP_01",
        "notes": "Verified in step 5 test suite."
    }
    save_res = client.post(f"/api/fill/active-learning/bottle/{bottle_id}/save", json=save_payload)
    assert save_res.status_code == 200

    zip_res = client.get("/api/fill/active-learning/export-dataset")
    assert zip_res.status_code == 200
    assert "application/zip" in zip_res.headers.get("content-type", "")
    assert len(zip_res.content) > 0
    logger.info("TEST 7 PASSED: Active learning candidate, save, and dataset zip export verified.")

    logger.info("=== TEST 8: Live WebSocket Stream ===")
    with client.websocket_connect("/ws/fill/stream") as ws:
        greeting = ws.receive_json()
        logger.info(f"WebSocket greeting: {greeting}")
        assert greeting["type"] == "connection_established"
        assert greeting["station"] == "fill_level"

        # Send ping
        ws.send_text(json.dumps({"command": "ping"}))
        pong = ws.receive_json()
        logger.info(f"WebSocket pong: {pong}")
        assert pong["type"] == "pong"

        # Send simulate
        ws.send_text(json.dumps({"command": "simulate"}))
        sim_ws = ws.receive_json()
        assert sim_ws["type"] == "inspection_cycle"
        assert sim_ws["station"] == "fill_level"
        logger.info(f"WebSocket simulated bottle: {sim_ws['data']['bottle_id']}")

        # Send get_stats
        ws.send_text(json.dumps({"command": "get_stats"}))
        stats_ws = ws.receive_json()
        assert stats_ws["type"] == "stats"
        assert stats_ws["station"] == "fill_level"
    logger.info("TEST 8 PASSED: Live WebSocket streaming verified.")

    logger.info("=== TEST 9: Symmetrical VRAM Management & Station Swapping ===")
    # 1. Unload Station 2
    res_unl2 = client.post("/api/fill/unload")
    assert res_unl2.status_code == 200
    assert res_unl2.json()["models_loaded"] is False
    assert client.get("/api/fill/health").json()["models_loaded"] is False
    logger.info("Station 2 unloaded successfully.")

    # 2. Check Station 1 (Cap) health and load Cap engine
    cap_health = client.get("/api/health").json()
    logger.info(f"Cap station initial health: {cap_health}")
    res_l1 = client.post("/api/load")
    assert res_l1.status_code == 200
    assert res_l1.json()["models_loaded"] is True
    assert client.get("/api/health").json()["models_loaded"] is True
    logger.info("Station 1 loaded successfully.")

    # 3. Unload Station 1
    res_unl1 = client.post("/api/unload")
    assert res_unl1.status_code == 200
    assert res_unl1.json()["models_loaded"] is False
    assert client.get("/api/health").json()["models_loaded"] is False
    logger.info("Station 1 unloaded successfully.")

    logger.info("TEST 9 PASSED: Symmetrical lazy loading and VRAM swapping 100% verified!")
    print("\n=======================================================")
    print("ALL PART 5 API, WEBSOCKET & VRAM SWAPPING TESTS PASSED!")
    print("=======================================================\n")


if __name__ == "__main__":
    run_tests()
