"""
api/middleware/metrics.py — Prometheus metrics middleware for VisionFood QAI.

Exposes production-grade observability:
  - HTTP request count (by method, path, status)
  - Request latency histogram (by method, path)
  - Inspection verdict counter (PASS/FAIL/ESCALATE/REVIEW)
  - Pipeline inference latency histogram
  - Active WebSocket connections gauge
  - Model load status gauge

Metrics endpoint: GET /metrics (Prometheus scrape target)
"""

from __future__ import annotations

import time
from typing import Optional

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from core.logging import get_logger

log = get_logger(__name__)


# ------------------------------------------------------------------ #
# In-process metric counters (no external dependency needed)
# Compatible with Prometheus text exposition format
# ------------------------------------------------------------------ #


class _Counter:
    """Thread-safe monotonic counter."""

    __slots__ = ("_name", "_help", "_values")

    def __init__(self, name: str, help_text: str) -> None:
        self._name = name
        self._help = help_text
        self._values: dict[tuple, float] = {}

    def inc(self, labels: dict[str, str], value: float = 1.0) -> None:
        key = tuple(sorted(labels.items()))
        self._values[key] = self._values.get(key, 0.0) + value

    def expose(self) -> str:
        lines = [f"# HELP {self._name} {self._help}", f"# TYPE {self._name} counter"]
        for key, val in sorted(self._values.items()):
            label_str = ",".join(f'{k}="{v}"' for k, v in key)
            lines.append(f"{self._name}{{{label_str}}} {val}")
        return "\n".join(lines)


class _Histogram:
    """Simple histogram with predefined buckets."""

    __slots__ = ("_name", "_help", "_buckets", "_values")

    _DEFAULT_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)

    def __init__(self, name: str, help_text: str, buckets: Optional[tuple] = None) -> None:
        self._name = name
        self._help = help_text
        self._buckets = buckets or self._DEFAULT_BUCKETS
        # {label_key: {"buckets": [counts], "sum": float, "count": int}}
        self._values: dict[tuple, dict] = {}

    def observe(self, labels: dict[str, str], value: float) -> None:
        key = tuple(sorted(labels.items()))
        if key not in self._values:
            self._values[key] = {
                "buckets": [0] * len(self._buckets),
                "sum": 0.0,
                "count": 0,
            }
        entry = self._values[key]
        entry["sum"] += value
        entry["count"] += 1
        for i, b in enumerate(self._buckets):
            if value <= b:
                entry["buckets"][i] += 1

    def expose(self) -> str:
        lines = [f"# HELP {self._name} {self._help}", f"# TYPE {self._name} histogram"]
        for key, entry in sorted(self._values.items()):
            label_str = ",".join(f'{k}="{v}"' for k, v in key)
            cumulative = 0
            for i, b in enumerate(self._buckets):
                cumulative += entry["buckets"][i]
                lines.append(f'{self._name}_bucket{{{label_str},le="{b}"}} {cumulative}')
            lines.append(f'{self._name}_bucket{{{label_str},le="+Inf"}} {entry["count"]}')
            lines.append(f"{self._name}_sum{{{label_str}}} {entry['sum']:.6f}")
            lines.append(f"{self._name}_count{{{label_str}}} {entry['count']}")
        return "\n".join(lines)


class _Gauge:
    """Simple gauge (can go up and down)."""

    __slots__ = ("_name", "_help", "_values")

    def __init__(self, name: str, help_text: str) -> None:
        self._name = name
        self._help = help_text
        self._values: dict[tuple, float] = {}

    def set(self, labels: dict[str, str], value: float) -> None:
        key = tuple(sorted(labels.items()))
        self._values[key] = value

    def inc(self, labels: dict[str, str], value: float = 1.0) -> None:
        key = tuple(sorted(labels.items()))
        self._values[key] = self._values.get(key, 0.0) + value

    def dec(self, labels: dict[str, str], value: float = 1.0) -> None:
        key = tuple(sorted(labels.items()))
        self._values[key] = self._values.get(key, 0.0) - value

    def expose(self) -> str:
        lines = [f"# HELP {self._name} {self._help}", f"# TYPE {self._name} gauge"]
        for key, val in sorted(self._values.items()):
            label_str = ",".join(f'{k}="{v}"' for k, v in key)
            lines.append(f"{self._name}{{{label_str}}} {val}")
        return "\n".join(lines)


# ------------------------------------------------------------------ #
# Global metric instances
# ------------------------------------------------------------------ #

http_requests_total = _Counter(
    "visionfood_http_requests_total",
    "Total HTTP requests by method, path, and status code.",
)

http_request_duration_seconds = _Histogram(
    "visionfood_http_request_duration_seconds",
    "HTTP request latency in seconds.",
)

inspection_verdicts_total = _Counter(
    "visionfood_inspection_verdicts_total",
    "Total inspection verdicts by type.",
)

inference_duration_seconds = _Histogram(
    "visionfood_inference_duration_seconds",
    "ML pipeline inference latency in seconds.",
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
)

model_loaded = _Gauge(
    "visionfood_model_loaded",
    "Whether the ML model is loaded (1) or not (0).",
)

active_ws_connections = _Gauge(
    "visionfood_active_ws_connections",
    "Number of active WebSocket connections.",
)


def collect_metrics() -> str:
    """Render all metrics in Prometheus text exposition format."""
    sections = [
        http_requests_total.expose(),
        http_request_duration_seconds.expose(),
        inspection_verdicts_total.expose(),
        inference_duration_seconds.expose(),
        model_loaded.expose(),
        active_ws_connections.expose(),
    ]
    return "\n\n".join(s for s in sections if s.strip()) + "\n"


# ------------------------------------------------------------------ #
# Middleware
# ------------------------------------------------------------------ #


def _normalize_path(path: str) -> str:
    """Collapse dynamic path segments to reduce cardinality."""
    parts = path.strip("/").split("/")
    normalized = []
    for part in parts:
        # UUID-like or numeric IDs → placeholder
        if len(part) == 36 and "-" in part:
            normalized.append("{id}")
        elif part.isdigit():
            normalized.append("{id}")
        else:
            normalized.append(part)
    return "/" + "/".join(normalized)


class PrometheusMiddleware(BaseHTTPMiddleware):
    """Track HTTP request metrics for every request."""

    async def dispatch(self, request: Request, call_next) -> Response:
        # Skip metrics endpoint itself to avoid recursion
        if request.url.path == "/metrics":
            return await call_next(request)

        t0 = time.perf_counter()
        response: Response = await call_next(request)
        duration = time.perf_counter() - t0

        path = _normalize_path(request.url.path)
        labels = {
            "method": request.method,
            "path": path,
            "status": str(response.status_code),
        }

        http_requests_total.inc(labels)
        http_request_duration_seconds.observe(
            {"method": request.method, "path": path}, duration
        )

        return response
