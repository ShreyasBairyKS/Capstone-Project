"""
api/main.py — FastAPI application entry point for VisionFood QAI.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from core.config import settings
from core.logging import get_logger, setup_logging

log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown tasks."""
    setup_logging(level=settings.LOG_LEVEL, log_format=settings.LOG_FORMAT)
    log.info("api_starting", tier=settings.TIER, device_id=settings.DEVICE_ID)

    # Initialise DB tables (run sync call in thread to avoid blocking event loop)
    from database.session import create_tables
    await asyncio.get_event_loop().run_in_executor(None, create_tables)
    log.info("database_tables_ready")

    # Load ONNX pipeline
    from inference.pipeline import EdgeInferencePipeline
    from api.dependencies import set_pipeline
    pipeline = EdgeInferencePipeline()
    try:
        pipeline.load_models()
    except Exception as exc:
        log.warning("pipeline_load_failed", error=str(exc))
        log.warning("api_running_without_models_use_dev_mode")
    set_pipeline(pipeline)

    yield

    log.info("api_shutdown")


app = FastAPI(
    title="VisionFood QAI",
    description="Intelligent quality inspection API for food & beverage manufacturing.",
    version=settings.APP_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ------------------------------------------------------------------ #
# Middleware (registered in reverse execution order)
# ------------------------------------------------------------------ #
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from api.middleware.auth import APIKeyMiddleware
from api.middleware.audit_logger import AuditLoggerMiddleware
from api.middleware.metrics import PrometheusMiddleware, collect_metrics

app.add_middleware(PrometheusMiddleware)
app.add_middleware(AuditLoggerMiddleware)
app.add_middleware(APIKeyMiddleware)

# ------------------------------------------------------------------ #
# Routers
# ------------------------------------------------------------------ #
from api.routers import inspection, analytics, reports, models, websocket

app.include_router(inspection.router)
app.include_router(analytics.router)
app.include_router(reports.router)
app.include_router(models.router)
app.include_router(websocket.router)


# ------------------------------------------------------------------ #
# Metrics
# ------------------------------------------------------------------ #
@app.get("/metrics", tags=["System"], include_in_schema=False)
async def prometheus_metrics():
    """Prometheus scrape endpoint — returns all metrics in text exposition format."""
    from fastapi.responses import PlainTextResponse
    return PlainTextResponse(
        collect_metrics(),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )


# ------------------------------------------------------------------ #
# Health check
# ------------------------------------------------------------------ #
@app.get("/health", tags=["System"])
async def health_check():
    """Liveness probe — exempt from API key auth."""
    from api.dependencies import get_pipeline
    from fastapi import HTTPException
    pipeline_loaded = False
    try:
        p = get_pipeline()
        pipeline_loaded = p._models_loaded
    except HTTPException:
        pass
    return {
        "status": "ok",
        "tier": settings.TIER,
        "device_id": settings.DEVICE_ID,
        "model_loaded": pipeline_loaded,
        "version": settings.APP_VERSION,
    }
