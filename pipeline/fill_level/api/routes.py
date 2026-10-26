"""
REST and WebSocket Routes for Station 2: Fill Level Inspection.
Mounted in pipeline/api/app.py under prefix /api/fill and direct WebSocket /ws/fill/stream.
"""
import io
import json
import logging
import time
from typing import Optional, List, Dict, Any, Set
import cv2
import numpy as np

from fastapi import (
    APIRouter,
    WebSocket,
    WebSocketDisconnect,
    UploadFile,
    File,
    Form,
    HTTPException,
    Query,
    Response,
)
from fastapi.concurrency import run_in_threadpool
from pydantic import ValidationError
from fastapi.responses import PlainTextResponse, StreamingResponse

from .schemas import (
    FillInspectResponse,
    FillActiveLearningCandidate,
    SaveFillAnnotationRequest,
    FillHealthResponse,
    FillConfigResponse,
    FillConfigUpdateRequest,
)
from .service import FillPipelineService

logger = logging.getLogger("pipeline.fill_level.api.routes")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

fill_router = APIRouter()


class FillConnectionManager:
    """Manages active WebSocket connections to Station 2."""

    def __init__(self):
        self.active_connections: Set[WebSocket] = set()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.add(websocket)
        logger.info(f"Station 2 WebSocket client connected. Active: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.discard(websocket)
        logger.info(f"Station 2 WebSocket client disconnected. Active: {len(self.active_connections)}")

    async def broadcast(self, message: Dict[str, Any]):
        dead = []
        payload = json.dumps(message)
        for ws in self.active_connections:
            try:
                await ws.send_text(payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)


ws_manager = FillConnectionManager()


# ── Health & Dynamic VRAM Allocation ──

@fill_router.get("/health", response_model=FillHealthResponse)
def get_health():
    """Reports Station 2 status and GPU VRAM allocation state."""
    svc = FillPipelineService.get_instance()
    return svc.get_health()


@fill_router.post("/load")
def load_engine():
    """Explicitly loads Fill Level models into GPU VRAM on station switch."""
    svc = FillPipelineService.get_instance()
    svc.ensure_engine_loaded()
    return {"status": "loaded", "models_loaded": True, "station": "Station 2: Fill Level"}


@fill_router.post("/unload")
def unload_engine():
    """Deallocates Fill Level models from VRAM and flushes CUDA memory."""
    svc = FillPipelineService.get_instance()
    svc.unload_engine()
    return {"status": "unloaded", "models_loaded": False, "station": "Station 2: Fill Level"}


# ── Runtime Quality-Gate Configuration (Station 2 only) ──

@fill_router.get("/config", response_model=FillConfigResponse)
def get_config():
    """Active Station 2 thresholds (independent of Station 1 /api/config)."""
    return FillPipelineService.get_instance().get_config()


@fill_router.post("/config", response_model=FillConfigResponse)
def update_config(req: FillConfigUpdateRequest):
    """Adjusts Station 2 thresholds live without restart."""
    svc = FillPipelineService.get_instance()
    try:
        return svc.update_config(req)
    except ValidationError as e:
        detail = "; ".join(f"{'.'.join(str(x) for x in err['loc'])}: {err['msg']}" for err in e.errors())
        raise HTTPException(status_code=422, detail=detail)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))


# ── Metrics & Defect Queries ──

@fill_router.get("/stats")
def get_stats(window_seconds: Optional[float] = None):
    """Returns real-time KPIs for Station 2."""
    svc = FillPipelineService.get_instance()
    return svc.get_stats(window_seconds)


def _clean_str(val: Any) -> Optional[str]:
    return val if isinstance(val, str) else None

def _clean_int(val: Any, default: int) -> int:
    return val if isinstance(val, int) else default

def _clean_bool(val: Any, default: bool) -> bool:
    return val if isinstance(val, bool) else default


@fill_router.get("/ratio-histogram")
def get_ratio_histogram(
    bins: int = Query(10, ge=2, le=50),
    window_seconds: Optional[float] = Query(None)
):
    """Real measured fill-ratio distribution (equal-width bins over 0-100%) for the dashboard histogram."""
    svc = FillPipelineService.get_instance()
    return svc.get_ratio_histogram(bins=bins, window_seconds=window_seconds)


@fill_router.get("/defects")
def get_defects(
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    verdict: Optional[str] = Query(None),
    fill_status: Optional[str] = Query(None),
    search: Optional[str] = Query(None),
    paginate: bool = Query(False)
):
    """Queries paginated defect inspections for Station 2."""
    svc = FillPipelineService.get_instance()
    return svc.get_defects(
        limit=_clean_int(limit, 50),
        offset=_clean_int(offset, 0),
        verdict=_clean_str(verdict),
        fill_status=_clean_str(fill_status),
        search=_clean_str(search),
        paginate=_clean_bool(paginate, False)
    )


@fill_router.get("/bottle/{bottle_id}")
def get_bottle_detail(bottle_id: str):
    """Deep-dive bottle inspection detail for S2 Replay Inspector."""
    svc = FillPipelineService.get_instance()
    data = svc.get_bottle_detail(bottle_id)
    if not data:
        raise HTTPException(status_code=404, detail=f"Bottle inspection '{bottle_id}' not found.")
    return data


@fill_router.get("/export-csv")
def export_csv(
    verdict: Optional[str] = Query(None),
    fill_status: Optional[str] = Query(None),
    search: Optional[str] = Query(None),
    limit: int = Query(10000, ge=1, le=50000)
):
    """Exports Station 2 audit records as RFC 4180 CSV."""
    svc = FillPipelineService.get_instance()
    csv_text = svc.export_csv(
        verdict=_clean_str(verdict),
        fill_status=_clean_str(fill_status),
        search=_clean_str(search),
        limit=_clean_int(limit, 10000)
    )
    return PlainTextResponse(
        content=csv_text,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=fill_level_audit.csv"}
    )


# ── Inspection Execution & Simulation ──

@fill_router.post("/simulate", response_model=FillInspectResponse)
async def simulate_cycle():
    """Triggers simulated inspection cycle and broadcasts event over WebSocket."""
    svc = FillPipelineService.get_instance()
    res = await run_in_threadpool(svc.run_simulated_cycle)
    stats = svc.get_stats()
    await ws_manager.broadcast({
        "type": "inspection_cycle",
        "station": "fill_level",
        "data": res.dict(),
        "stats": stats.dict()
    })
    return res


@fill_router.post("/inspect", response_model=FillInspectResponse)
async def inspect_upload(
    file: UploadFile = File(...),
    bottle_id: Optional[str] = Form(None),
    bottle_id_query: Optional[str] = Query(None, alias="bottle_id")
):
    """Inspects an uploaded profile bottle image."""
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Invalid image file format.")

    target_bottle_id = bottle_id or bottle_id_query
    svc = FillPipelineService.get_instance()
    res = await run_in_threadpool(svc.inspect_frame, frame, bottle_id=target_bottle_id)
    stats = svc.get_stats()
    await ws_manager.broadcast({
        "type": "inspection_cycle",
        "station": "fill_level",
        "data": res.dict(),
        "stats": stats.dict()
    })
    return res


# ── Active Learning ──

@fill_router.get("/active-learning/candidates", response_model=List[FillActiveLearningCandidate])
def get_candidates(
    uncertain_only: bool = Query(True),
    search: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=200)
):
    """Queries borderline cases for Station 3 Active Learning queue."""
    svc = FillPipelineService.get_instance()
    return svc.get_active_learning_candidates(
        uncertain_only=_clean_bool(uncertain_only, True),
        search=_clean_str(search),
        limit=_clean_int(limit, 50)
    )


@fill_router.post("/active-learning/annotate")
def save_annotation(req: SaveFillAnnotationRequest):
    """Commits ground-truth adjustments from S3 Active Learning."""
    svc = FillPipelineService.get_instance()
    return svc.save_active_learning_annotation(req)


@fill_router.get("/active-learning/export-zip")
def export_dataset():
    """Downloads standardized YOLOv11 dataset zip archive for fill models."""
    svc = FillPipelineService.get_instance()
    zip_bytes = svc.export_dataset_zip()
    return StreamingResponse(
        io.BytesIO(zip_bytes),
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=fill_dataset.zip"}
    )


# ── Live WebSocket Endpoint ──

@fill_router.websocket("/ws/fill/stream")
async def websocket_fill_stream(websocket: WebSocket):
    """
    Dedicated high-throughput WebSocket stream for Station 2 (Fill Level).
    Broadcasts live video overlays and real-time KPI updates.
    """
    await ws_manager.connect(websocket)
    svc = FillPipelineService.get_instance()

    # Dynamic auto-load on first dashboard connection (may trigger a first-time TensorRT compile)
    try:
        await run_in_threadpool(svc.ensure_engine_loaded)
    except Exception as e:
        logger.error(f"Station 2 engine failed to load on WebSocket connect: {e}")
        ws_manager.disconnect(websocket)
        await websocket.close(code=1011, reason="Fill Level engine failed to load")
        return
    stats = svc.get_stats()
    hw = svc.get_health().hardware_runtime

    await websocket.send_text(json.dumps({
        "type": "connection_established",
        "station": "fill_level",
        "hardware": hw,
        "stats": stats.dict()
    }))

    try:
        while True:
            text = await websocket.receive_text()
            try:
                data = json.loads(text)
                cmd = data.get("command", "")
                if cmd in ("simulate", "simulate_cycle"):
                    res = await run_in_threadpool(svc.run_simulated_cycle)
                    st = svc.get_stats()
                    await ws_manager.broadcast({
                        "type": "inspection_cycle",
                        "station": "fill_level",
                        "data": res.dict(),
                        "stats": st.dict()
                    })
                elif cmd == "ping":
                    await websocket.send_text(json.dumps({"type": "pong", "time": time.time()}))
                elif cmd == "get_stats":
                    st = svc.get_stats()
                    await websocket.send_text(json.dumps({"type": "stats", "station": "fill_level", "stats": st.dict()}))
            except Exception as e:
                logger.error(f"Error handling WebSocket message in Station 2: {e}")
                try:
                    await websocket.send_text(json.dumps({"type": "error", "station": "fill_level", "message": str(e)}))
                except Exception:
                    pass
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
    except Exception as e:
        logger.warning(f"Station 2 WebSocket connection error: {e}")
        ws_manager.disconnect(websocket)
