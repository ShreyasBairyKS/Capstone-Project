"""
FastAPI Application Entrypoint for VisionQAI API Layer.
Configures CORS, lifespan events, REST routes, WebSockets, and static frontend hosting.
"""
import logging
from pathlib import Path

import pipeline.env_loader  # noqa: F401  (loads <repo>/.env before any engine/service reads settings)
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse

from pipeline.api.routes import router, websocket_stream_endpoint
from pipeline.api.pipeline_service import PipelineService
from pipeline.fill_level.api.routes import fill_router, websocket_fill_stream
from pipeline.fill_level.api.service import FillPipelineService

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s")
logger = logging.getLogger("pipeline.api.app")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler: Warms up pipeline on startup, terminates workers on exit."""
    logger.info("Starting VisionQAI Automated Inspection API Service...")
    svc = PipelineService.get_instance()
    hw = svc.inference_engine.hardware_desc if svc.inference_engine else 'Standby (Unloaded)'
    logger.info(f"Cap inference engine online on: {hw}")

    fill_svc = FillPipelineService.get_instance()
    fill_hw = fill_svc.engine.hardware_desc if fill_svc.engine else 'Standby (Unloaded)'
    logger.info(f"Fill level inference engine status: {fill_hw}")

    yield
    logger.info("Shutting down API Service...")
    try:
        svc.shutdown()
    except Exception as e:
        logger.warning(f"Error shutting down PipelineService: {e}")
    try:
        fill_svc.shutdown()
    except Exception as e:
        logger.warning(f"Error shutting down FillPipelineService: {e}")
    logger.info("API Service shutdown complete.")


def create_app() -> FastAPI:
    """Factory creating configured FastAPI application for VisionQAI."""
    app = FastAPI(
        title="VisionQAI Automated Bottle Cap Defect Detection API",
        description="High-throughput manufacturing defect detection, multi-camera quality gating, and dynamic replay service.",
        version="2.0.0",
        docs_url="/visionqai/docs",
        redoc_url="/visionqai/redoc",
        lifespan=lifespan
    )

    # Enable full CORS for frontend dashboard
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Direct redirect from root / and /visionqai to /visionqai/
    @app.get("/", include_in_schema=False)
    async def redirect_root_to_visionqai():
        return RedirectResponse(url="/visionqai/")

    @app.get("/visionqai", include_in_schema=False)
    async def redirect_slug_to_slash():
        return RedirectResponse(url="/visionqai/")

    @app.get("/docs", include_in_schema=False)
    async def redirect_docs():
        return RedirectResponse(url="/visionqai/docs")

    # 1. Mount Station 1 (Cap) REST routes under /api
    app.include_router(router, prefix="/api", tags=["Station 1: Cap Inspection API"])

    # 2. Mount Station 2 (Fill Level) REST routes under /api/fill
    app.include_router(fill_router, prefix="/api/fill", tags=["Station 2: Fill Level Inspection API"])

    # 3. Direct WebSocket stream endpoints
    @app.websocket("/ws/stream")
    async def direct_ws_cap(websocket: WebSocket):
        await websocket_stream_endpoint(websocket)

    @app.websocket("/ws/fill/stream")
    async def direct_ws_fill(websocket: WebSocket):
        await websocket_fill_stream(websocket)

    # 4. Mount production built React frontend if dist exists
    frontend_dist = Path(__file__).resolve().parent.parent / "frontend" / "dist"
    if frontend_dist.exists():
        logger.info(f"Mounting VisionQAI frontend production assets from {frontend_dist}")
        # Mount under /visionqai so URL path reflects brand
        app.mount("/visionqai", StaticFiles(directory=str(frontend_dist), html=True), name="visionqai_frontend")
        # Fallback mount at root for direct assets
        app.mount("/", StaticFiles(directory=str(frontend_dist), html=True), name="root_fallback")

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("pipeline.api.app:app", host="0.0.0.0", port=8000, reload=False)
