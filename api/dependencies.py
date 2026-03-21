"""
api/dependencies.py — FastAPI dependency injection helpers.

Provides reusable `Depends(...)` wrappers for:
  - Database sessions
  - Inference pipeline singleton
  - API key authentication
"""

from __future__ import annotations

from typing import Generator

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import APIKeyHeader

from core.config import settings
from database.session import SessionLocal
from inference.pipeline import EdgeInferencePipeline

# ------------------------------------------------------------------ #
# Singleton pipeline — module-level, shared across requests
# ------------------------------------------------------------------ #
_pipeline: EdgeInferencePipeline | None = None


def get_pipeline() -> EdgeInferencePipeline:
    """Return the singleton inference pipeline (loaded at startup)."""
    if _pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference pipeline not yet initialised.",
        )
    return _pipeline


def set_pipeline(p: EdgeInferencePipeline) -> None:
    """Called from lifespan startup to register the loaded pipeline."""
    global _pipeline
    _pipeline = p


# ------------------------------------------------------------------ #
# Database session
# ------------------------------------------------------------------ #

def get_db() -> Generator:
    """
    Yield a SQLAlchemy Session, ensuring it is closed after the request.
    Usage: db: Session = Depends(get_db)
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ------------------------------------------------------------------ #
# API key authentication
# ------------------------------------------------------------------ #

_API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


def verify_api_key(api_key: str | None = Security(_API_KEY_HEADER)) -> str:
    """
    Validate the X-API-Key header against the configured secret.

    Returns the key string on success.
    Raises HTTP 401 if missing or invalid.
    """
    if not api_key or api_key != settings.API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key.",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    return api_key
