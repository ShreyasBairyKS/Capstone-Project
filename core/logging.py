"""
core/logging.py — Structured JSON logging for VisionFood QAI.

Usage:
    from core.logging import get_logger, setup_logging

    setup_logging(level="INFO", log_format="json")
    log = get_logger(__name__)
    log.info("inspection_complete", verdict="PASS", latency_ms=42.1)
"""

import logging
import sys
from typing import Literal

import structlog


def setup_logging(
    level: str = "INFO",
    log_format: Literal["json", "text"] = "json",
) -> None:
    """Configure structlog with either JSON (production) or console (dev) output."""

    log_level = getattr(logging, level.upper(), logging.INFO)

    shared_processors = [
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]

    if log_format == "json":
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=shared_processors + [renderer],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=log_level,
    )

    # Silence noisy third-party loggers
    for noisy_logger in ("uvicorn.access", "multipart"):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Return a named structlog logger bound to the given module name."""
    return structlog.get_logger(name)
