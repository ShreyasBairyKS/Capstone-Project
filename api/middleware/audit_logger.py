"""
api/middleware/audit_logger.py — JSONL audit trail middleware.

Writes one audit line per request to the audit log file configured in settings.
Captures: timestamp, method, path, status_code, latency_ms, api_key (masked).
"""

from __future__ import annotations

import atexit
import json
import time
from pathlib import Path

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from core.config import settings
from core.logging import get_logger

log = get_logger(__name__)

_AUDIT_PATH = Path(settings.AUDIT_LOG_PATH)


def _mask_key(key: str | None) -> str:
    """Show only last 4 chars of API key to aid debugging without exposing secrets."""
    if not key:
        return "none"
    return "****" + key[-4:] if len(key) > 4 else "****"


class AuditLoggerMiddleware(BaseHTTPMiddleware):
    """
    Appends a structured JSON audit record for every HTTP request.

    The audit file is written in JSONL format (one JSON object per line)
    so it can be streamed cheaply into any log aggregation platform.
    The file handle is kept open and line-buffered for efficiency.
    """

    def __init__(self, app, audit_path: Path = _AUDIT_PATH) -> None:
        super().__init__(app)
        self._audit_path = audit_path
        self._audit_path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self._audit_path, "a", encoding="utf-8", buffering=1)  # line-buffered
        atexit.register(self._fh.close)

    async def dispatch(self, request: Request, call_next) -> Response:
        t0 = time.perf_counter()
        response: Response = await call_next(request)
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)

        record = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "method": request.method,
            "path": request.url.path,
            "status": response.status_code,
            "latency_ms": latency_ms,
            "client": request.client.host if request.client else None,
            "api_key": _mask_key(request.headers.get("X-API-Key")),
        }

        try:
            self._fh.write(json.dumps(record) + "\n")
        except OSError as exc:
            log.warning("audit_log_write_failed", error=str(exc))

        return response
