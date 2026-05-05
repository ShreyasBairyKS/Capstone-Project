"""
api/dependencies.py — FastAPI dependency injection helpers.

Provides reusable `Depends(...)` wrappers for:
  - Database sessions (SQLAlchemy + Motor)
  - Inference pipeline singleton
  - API key authentication
  - Role-based access control (RBAC)
"""

from __future__ import annotations

from typing import Generator

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import APIKeyHeader
from sqlalchemy.orm import Session

from core.config import settings
from database.session import get_motor_db  # re-export Motor dep

# ------------------------------------------------------------------ #
# Singleton pipeline — module-level, shared across requests
# ------------------------------------------------------------------ #
_pipeline = None


def get_pipeline():
    """Return the singleton inference pipeline (loaded at startup)."""
    from inference.pipeline import EdgeInferencePipeline  # lazy import
    if _pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference pipeline not yet initialised.",
        )
    return _pipeline


def set_pipeline(p) -> None:
    """Called from lifespan startup to register the loaded pipeline."""
    global _pipeline
    _pipeline = p


# ------------------------------------------------------------------ #
# Database session
# ------------------------------------------------------------------ #

def get_db() -> Generator[Session, None, None]:
    """
    Yield a SQLAlchemy Session, ensuring it is closed after the request.
    Usage: db: Session = Depends(get_db)
    """
    from database.session import SessionLocal  # lazy import to avoid circular deps
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Motor dependency is imported from database.session and re-exported here
# so routers only need: from api.dependencies import get_motor_db
__all__ = ["get_db", "get_motor_db", "get_pipeline", "set_pipeline", "verify_api_key", "require_role"]


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


# ------------------------------------------------------------------ #
# Role-based access control (RBAC)
# ------------------------------------------------------------------ #

# Role hierarchy — higher index = more privileged
_ROLE_HIERARCHY: list[str] = ["viewer", "operator", "supervisor", "admin"]


def require_role(minimum_role: str):
    """
    Dependency factory that returns a FastAPI dependency enforcing a minimum role.

    Roles (ascending privilege): viewer < operator < supervisor < admin

    The role is extracted from the ``X-User-Role`` request header. In production
    this header is set by the auth middleware after JWT validation. In development
    it defaults to ``"admin"`` if the header is absent.

    Usage:
        @router.post("/", dependencies=[Depends(require_role("supervisor"))])
        async def create(...): ...

        # Or with injection if you need the role value:
        async def create(_role: str = Depends(require_role("supervisor"))): ...

    Args:
        minimum_role: The least-privileged role that may access the endpoint.

    Returns:
        A callable FastAPI dependency that raises HTTP 403 if the role is insufficient.
    """
    if minimum_role not in _ROLE_HIERARCHY:
        raise ValueError(
            f"Unknown minimum_role '{minimum_role}'. Must be one of {_ROLE_HIERARCHY}."
        )
    min_index = _ROLE_HIERARCHY.index(minimum_role)

    from fastapi import Header

    async def _check_role(x_user_role: str | None = Header(default=None, alias="X-User-Role")) -> str:
        # Fallback to "admin" in dev so the API is usable without a full auth stack
        role = x_user_role or "admin"
        if role not in _ROLE_HIERARCHY:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Unrecognised role '{role}'.",
            )
        if _ROLE_HIERARCHY.index(role) < min_index:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient role. Required: '{minimum_role}', got: '{role}'.",
            )
        return role

    return _check_role
