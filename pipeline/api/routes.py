"""
FastAPI Routes for Part 6: Backend Service & API Layer.
Implements REST endpoints and WebSockets for inspection, telemetry,
dynamic image replay, and runtime configuration.
"""
import json
import logging
from typing import List, Optional
import cv2
import numpy as np
from fastapi import APIRouter, UploadFile, File, Form, Query, HTTPException, Response, WebSocket, WebSocketDisconnect

from pipeline.api.schemas import (
    InspectResponse,
    ConfigUpdateRequest,
    ConfigResponse,
    DefectItem
)
from pipeline.audit.schemas import KPISummary
from pipeline.api.pipeline_service import PipelineService

logger = logging.getLogger("pipeline.api.routes")
router = APIRouter()


def get_service() -> PipelineService:
    return PipelineService.get_instance()


# -----------------------------------------------------------------------------
# REST Endpoints
# -----------------------------------------------------------------------------

@router.get("/health")
def health_check():
    """System health check and hardware acceleration profile."""
    svc = get_service()
    return {
        "status": "online",
        "service": "VisionQAI Industrial Inspection Pipeline",
        "hardware": svc.inference_engine.hardware_desc,
        "classes": list(svc.inference_engine.model.names.values())
    }


@router.post("/inspect/image", response_model=InspectResponse)
async def inspect_images(
    files: List[UploadFile] = File(..., description="One or more bottle view images"),
    camera_ids: Optional[str] = Form(None, description="Comma-separated camera identifiers (e.g. camera_front,camera_rear)"),
    bottle_id: Optional[str] = Form(None, description="Optional custom bottle inspection ID")
):
    """
    Inspect uploaded images across one or multiple camera angles.
    Executes Ingestion -> Batched Inference -> Quality Gate -> Defect Audit -> Visualizer.
    """
    svc = get_service()
    cam_id_list = [c.strip() for c in camera_ids.split(",")] if camera_ids else []

    frames = {}
    default_cams = ["camera_front", "camera_rear", "camera_top", "camera_side"]

    for i, file in enumerate(files):
        content = await file.read()
        nparr = np.frombuffer(content, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=400, detail=f"Could not decode image: {file.filename}")

        if i < len(cam_id_list):
            cam_name = cam_id_list[i]
        elif i < len(default_cams):
            cam_name = default_cams[i]
        else:
            cam_name = f"camera_{i+1}"

        frames[cam_name] = img

    if not frames:
        raise HTTPException(status_code=400, detail="No valid images provided.")

    return svc.inspect_frames(frames=frames, bottle_id=bottle_id, render_base64=True)


@router.post("/simulate", response_model=InspectResponse)
def simulate_cycle():
    """Triggers an inspection cycle with simulated conveyor frames for demo and testing."""
    svc = get_service()
    return svc.run_simulated_cycle()


@router.get("/stats", response_model=KPISummary)
def get_stats(window_seconds: Optional[float] = Query(None, description="Trailing time window in seconds")):
    """Returns real-time industrial KPIs: total inspected, pass/reject rates, and defect distribution."""
    svc = get_service()
    return svc.get_stats(window_seconds)


@router.get("/defects", response_model=List[DefectItem])
def get_defects(
    limit: int = Query(50, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    verdict: Optional[str] = Query("REJECT", description="Filter by verdict: REJECT, UNCERTAIN, or PASS"),
    defect_type: Optional[str] = Query(None, description="Filter by defect class name (e.g. damaged_cap)")
):
    """Returns paginated historical defect events for auditing."""
    svc = get_service()
    return svc.get_defects(limit=limit, offset=offset, verdict=verdict, defect_type=defect_type)


@router.get("/bottles/{bottle_id}")
def get_bottle(bottle_id: str):
    """Retrieves full inspection metadata, detection contours, and file manifests for a bottle."""
    svc = get_service()
    detail = svc.get_bottle_detail(bottle_id)
    if not detail:
        raise HTTPException(status_code=404, detail=f"Bottle {bottle_id} not found.")
    return detail


@router.get("/defects/{bottle_id}")
def get_defect_bottle(bottle_id: str):
    """Alias for /bottles/{bottle_id}."""
    return get_bottle(bottle_id)


@router.get("/bottles/{bottle_id}/image/{camera_id}")
def get_bottle_image(
    bottle_id: str,
    camera_id: str,
    render: bool = Query(False, description="True for dynamic transparent overlay; False for pristine raw image")
):
    """
    Serves camera image for a specific bottle.
    Supports on-demand dynamic replay rendering from clean raw storage.
    """
    svc = get_service()
    res = svc.get_camera_image_bytes(bottle_id, camera_id, render=render)
    if not res:
        raise HTTPException(status_code=404, detail=f"Image for bottle {bottle_id} ({camera_id}) not found.")
    img_bytes, media_type = res
    return Response(content=img_bytes, media_type=media_type)


@router.get("/config", response_model=ConfigResponse)
def get_configuration():
    """Retrieves active runtime thresholds and storage retention configuration."""
    svc = get_service()
    return svc.get_config()


@router.post("/config", response_model=ConfigResponse)
def update_configuration(req: ConfigUpdateRequest):
    """Updates operating thresholds dynamically without restarting service."""
    svc = get_service()
    return svc.update_config(req)


# -----------------------------------------------------------------------------
# WebSocket Stream Endpoint
# -----------------------------------------------------------------------------

@router.websocket("/ws/stream")
async def websocket_stream_endpoint(websocket: WebSocket):
    """
    Full-duplex WebSocket stream for real-time frontend integration.
    Pushes synchronized multi-camera frames, HUD overlays, and decision telemetry.
    Supports client commands ('simulate', 'ping', 'get_stats').
    """
    await websocket.accept()
    svc = get_service()
    logger.info("WebSocket client connected to /ws/stream")

    try:
        # Send initial connection greeting with hardware and stats
        init_payload = {
            "type": "connection_established",
            "hardware": svc.inference_engine.hardware_desc,
            "stats": svc.get_stats().model_dump()
        }
        await websocket.send_text(json.dumps(init_payload))

        while True:
            # Receive client message/command
            data_str = await websocket.receive_text()
            try:
                msg = json.loads(data_str)
                cmd = msg.get("command", "")
            except Exception:
                cmd = data_str.strip().lower()

            if cmd in ("simulate", "simulate_cycle"):
                # Run inspection and broadcast full telemetry + base64 views
                result = svc.run_simulated_cycle()
                stats = svc.get_stats().model_dump()
                response = {
                    "type": "inspection_cycle",
                    "data": result.model_dump(),
                    "stats": stats
                }
                await websocket.send_text(json.dumps(response))

            elif cmd == "get_stats":
                stats = svc.get_stats().model_dump()
                await websocket.send_text(json.dumps({"type": "stats", "stats": stats}))

            elif cmd == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))

            else:
                await websocket.send_text(json.dumps({"type": "error", "message": f"Unknown command: {cmd}"}))

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected.")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
