"""
Part 6: Backend Service & API Layer
FastAPI application entrypoint, REST routers, and WebSocket streams.
"""
from pipeline.api.app import app, create_app
from pipeline.api.pipeline_service import PipelineService

__all__ = ["app", "create_app", "PipelineService"]
