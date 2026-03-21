"""
api/middleware/auth.py — API key middleware (process-level guard).

Every request must carry a valid X-API-Key header.
The health endpoint is exempted so load balancers can probe freely.
"""

from __future__ import annotations

from fastapi import Request, Response, status
from starlette.middleware.base import BaseHTTPMiddleware

from core.config import settings
from core.logging import get_logger

log = get_logger(__name__)

# Paths that do NOT require authentication
_PUBLIC_PATHS = {"/health", "/docs", "/redoc", "/openapi.json"}


class APIKeyMiddleware(BaseHTTPMiddleware):
    """
    Reject requests with a missing or invalid X-API-Key header.
    Returns 401 JSON immediately without hitting any route handler.
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        # Allow public endpoints without authentication
        if request.url.path in _PUBLIC_PATHS:
            return await call_next(request)

        api_key = request.headers.get("X-API-Key")
        if not api_key or api_key != settings.API_KEY:
            log.warning(
                "auth_rejected",
                path=request.url.path,
                method=request.method,
                client=request.client.host if request.client else "unknown",
            )
            return Response(
                content='{"detail":"Invalid or missing API key."}',
                status_code=status.HTTP_401_UNAUTHORIZED,
                media_type="application/json",
                headers={"WWW-Authenticate": "ApiKey"},
            )

        return await call_next(request)
