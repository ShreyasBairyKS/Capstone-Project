"""
FastAPI Application Entrypoint for VisionQAI API Layer.
Configures CORS, lifespan events, REST routes, WebSockets, and static frontend hosting.
"""
import logging
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse

from pipeline.api.routes import router
from pipeline.api.pipeline_service import PipelineService

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s")
logger = logging.getLogger("pipeline.api.app")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler: Warms up pipeline on startup, terminates workers on exit."""
    logger.info("Starting VisionQAI Automated Inspection API Service...")
    svc = PipelineService.get_instance()
    logger.info(f"Inference engine online on: {svc.inference_engine.hardware_desc}")
    yield
    logger.info("Shutting down API Service...")
    svc.shutdown()
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

    # 1. Mount API REST routes under /api
    app.include_router(router, prefix="/api", tags=["Pipeline API"])

    # 2. Mount WebSocket router at root
    app.include_router(router, tags=["Direct Routes"])

    # 3. Mount production built React frontend if dist exists
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
