"""
Comprehensive Verification Suite for Part 6: Backend Service & API Layer.
Validates:
1. System health check and runtime hardware diagnostics (GET /api/health).
2. Dynamic threshold and configuration adjustment without restart (GET/POST /api/config).
3. Multi-view image upload and full pipeline inspection (POST /api/inspect/image).
4. Real-time manufacturing KPIs and defect audit event querying (GET /api/stats, GET /api/defects).
5. Isolated bottle retrieval and dynamic on-demand replay rendering (GET /api/bottles/...).
6. Full-duplex WebSocket stream communication (WebSocket /ws/stream).
"""
import os
import sys
import io
import json
import logging
from pathlib import Path
import cv2
import numpy as np
from fastapi.testclient import TestClient

# Ensure pipeline root is in sys.path
pipeline_root = Path(__file__).resolve().parent
project_root = pipeline_root.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from pipeline.api.app import app
from pipeline.api.pipeline_service import PipelineService

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s")
logger = logging.getLogger("test_step6")


def test_part6_api():
    print("=" * 75)
    print("  STEP 6 VERIFICATION: FASTAPI REST & WEBSOCKETS BACKEND LAYER")
    print("=" * 75)

    client = TestClient(app)

    # -------------------------------------------------------------------------
    # [0/6] Testing Static Frontend Dashboard (GET /)
    print("\n[0/6] Testing Static Frontend Dashboard (GET /)...")
    resp_home = client.get("/")
    assert resp_home.status_code == 200
    assert "Automated Bottle Cap Defect Detection" in resp_home.text
    print("  ✓ Frontend SPA index.html served cleanly at root /")

    # [1/6] System Health & Hardware Diagnostics
    # -------------------------------------------------------------------------
    print("\n[1/6] Testing GET /api/health...")
    resp = client.get("/api/health")
    assert resp.status_code == 200, f"Health check failed: {resp.text}"
    health_data = resp.json()
    assert health_data["status"] == "online"
    assert "hardware" in health_data
    assert len(health_data["classes"]) == 6
    print(f"  ✓ Service status : {health_data['status']}")
    print(f"  ✓ Active hardware: {health_data['hardware']}")
    print(f"  ✓ Classes (6)    : {health_data['classes']}")

    # -------------------------------------------------------------------------
    # [2/6] Dynamic Configuration & Threshold Updates
    # -------------------------------------------------------------------------
    print("\n[2/6] Testing GET & POST /api/config...")
    resp_cfg = client.get("/api/config")
    assert resp_cfg.status_code == 200
    cfg_orig = resp_cfg.json()
    print(f"  ✓ Initial thresholds: Pass={cfg_orig['pass_conf_threshold']}, Defect={cfg_orig['defect_conf_threshold']}")

    # Update thresholds dynamically
    new_cfg_payload = {
        "pass_conf_threshold": 0.82,
        "defect_conf_threshold": 0.48,
        "mask_alpha": 0.45
    }
    resp_update = client.post("/api/config", json=new_cfg_payload)
    assert resp_update.status_code == 200
    cfg_updated = resp_update.json()
    assert cfg_updated["pass_conf_threshold"] == 0.82
    assert cfg_updated["defect_conf_threshold"] == 0.48
    assert cfg_updated["mask_alpha"] == 0.45
    print(f"  ✓ Updated live thresholds without restart: Pass={cfg_updated['pass_conf_threshold']}, Defect={cfg_updated['defect_conf_threshold']}, Alpha={cfg_updated['mask_alpha']}")

    # Revert to standard thresholds
    client.post("/api/config", json={"pass_conf_threshold": 0.75, "defect_conf_threshold": 0.45, "mask_alpha": 0.40})

    # -------------------------------------------------------------------------
    # [3/6] Multi-View Inspection via REST Upload (POST /api/inspect/image)
    # -------------------------------------------------------------------------
    print("\n[3/6] Testing POST /api/inspect/image with Multi-Camera Upload...")
    # Create two synthetic JPEG test buffers
    f1 = np.full((400, 400, 3), (80, 80, 80), dtype=np.uint8)
    f2 = np.full((400, 400, 3), (80, 80, 80), dtype=np.uint8)
    _, buf1 = cv2.imencode(".jpg", f1)
    _, buf2 = cv2.imencode(".jpg", f2)

    files = [
        ("files", ("camera_front.jpg", buf1.tobytes(), "image/jpeg")),
        ("files", ("camera_rear.jpg", buf2.tobytes(), "image/jpeg"))
    ]
    data = {
        "camera_ids": "camera_front,camera_rear",
        "bottle_id": "BOTTLE_API_TEST_001"
    }

    resp_inspect = client.post("/api/inspect/image", files=files, data=data)
    assert resp_inspect.status_code == 200, f"Inspection failed: {resp_inspect.text}"
    inspect_data = resp_inspect.json()

    assert inspect_data["bottle_id"] == "BOTTLE_API_TEST_001"
    assert "global_verdict" in inspect_data
    assert "is_compliant" in inspect_data
    assert "total_latency_ms" in inspect_data
    assert "views" in inspect_data
    assert "camera_front" in inspect_data["views"]
    assert "camera_rear" in inspect_data["views"]
    # Check base64 rendered image
    b64_front = inspect_data["views"]["camera_front"]["annotated_image_base64"]
    assert b64_front is not None and b64_front.startswith("data:image/jpeg;base64,")

    print(f"  ✓ Bottle ID      : {inspect_data['bottle_id']}")
    print(f"  ✓ Global Verdict : {inspect_data['global_verdict']}")
    print(f"  ✓ Latency        : {inspect_data['total_latency_ms']} ms")
    print(f"  ✓ Multi-views (2): {list(inspect_data['views'].keys())} with live base64 overlays")

    # -------------------------------------------------------------------------
    # [4/6] Telemetry Analytics & Defect Queries (GET /api/stats, GET /api/defects)
    # -------------------------------------------------------------------------
    print("\n[4/6] Testing GET /api/stats & GET /api/defects...")
    resp_stats = client.get("/api/stats")
    assert resp_stats.status_code == 200
    stats_data = resp_stats.json()
    assert stats_data["total_inspected"] >= 1
    print(f"  ✓ Rolling Stats  : Inspected={stats_data['total_inspected']} | Pass Rate={stats_data['pass_rate_pct']}% | Reject Rate={stats_data['reject_rate_pct']}%")

    resp_defects = client.get("/api/defects?limit=10")
    assert resp_defects.status_code == 200
    defects_data = resp_defects.json()
    assert isinstance(defects_data, list)
    print(f"  ✓ Defects history: Query returned {len(defects_data)} defect record(s)")

    # -------------------------------------------------------------------------
    # [5/6] Isolated Bottle Detail & Dynamic On-Demand Replay Rendering
    # -------------------------------------------------------------------------
    print("\n[5/6] Testing GET /api/bottles/{bottle_id} & Dynamic Replay Image...")
    resp_bottle = client.get(f"/api/bottles/{inspect_data['bottle_id']}")
    assert resp_bottle.status_code == 200
    bottle_detail = resp_bottle.json()
    assert "record" in bottle_detail
    assert "manifest" in bottle_detail
    print(f"  ✓ Bottle detail retrieved for: {inspect_data['bottle_id']}")

    # Test raw image serving (render=false)
    resp_raw = client.get(f"/api/bottles/{inspect_data['bottle_id']}/image/camera_front?render=false")
    assert resp_raw.status_code == 200
    assert resp_raw.headers["content-type"] == "image/jpeg"
    assert resp_raw.content[:3] == b"\xff\xd8\xff"  # Valid JPEG header
    print(f"  ✓ Raw unannotated JPEG image served ({len(resp_raw.content)} bytes)")

    # Test dynamic on-demand replay rendering (render=true)
    resp_render = client.get(f"/api/bottles/{inspect_data['bottle_id']}/image/camera_front?render=true")
    assert resp_render.status_code == 200
    assert resp_render.headers["content-type"] == "image/jpeg"
    assert resp_render.content[:3] == b"\xff\xd8\xff"
    print(f"  ✓ Dynamic on-demand replay rendered JPEG served ({len(resp_render.content)} bytes)")

    # -------------------------------------------------------------------------
    # [6/6] Full-Duplex WebSocket Stream (WebSocket /ws/stream)
    # -------------------------------------------------------------------------
    print("\n[6/6] Testing WebSocket Stream (/ws/stream)...")
    with client.websocket_connect("/ws/stream") as ws:
        # 1. Receive initial connection greeting
        init_msg = ws.receive_json()
        assert init_msg["type"] == "connection_established"
        print(f"  ✓ WS Connection established: Hardware={init_msg['hardware']}")

        # 2. Test ping / pong
        ws.send_text(json.dumps({"command": "ping"}))
        pong_msg = ws.receive_json()
        assert pong_msg["type"] == "pong"
        print("  ✓ WS Ping/Pong heartbeat verified")

        # 3. Test simulated cycle trigger over WebSocket
        ws.send_text(json.dumps({"command": "simulate"}))
        sim_msg = ws.receive_json()
        assert sim_msg["type"] == "inspection_cycle"
        assert "data" in sim_msg
        assert "stats" in sim_msg
        assert "bottle_id" in sim_msg["data"]
        assert "views" in sim_msg["data"]
        print(f"  ✓ WS Inspection cycle broadcast: Bottle={sim_msg['data']['bottle_id']} | Verdict={sim_msg['data']['global_verdict']}")

    print("\n" + "=" * 75)
    print("  >>> STEP 6 VERIFICATION PASSED: FastAPI REST & WebSockets Backend is 100% Operational <<<")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    test_part6_api()
