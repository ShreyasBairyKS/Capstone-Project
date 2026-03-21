"""
core/messaging.py — Redis Stream wrappers for live inspection event publishing.

Used by the inference pipeline to push results to the dashboard WebSocket
without coupling the pipeline directly to the API layer.
"""

import json
from typing import Any

import redis.asyncio as aioredis

from core.config import settings
from core.logging import get_logger

log = get_logger(__name__)


async def get_redis_client() -> aioredis.Redis:
    """Create and return an async Redis client."""
    return await aioredis.from_url(settings.REDIS_URL, decode_responses=True)


async def publish_inspection_event(client: aioredis.Redis, payload: dict[str, Any]) -> None:
    """
    Publish an inspection result to the live Redis Stream.

    The WebSocket router consumes this stream and pushes events to all
    connected dashboard clients.
    """
    try:
        await client.xadd(
            settings.REDIS_LIVE_STREAM,
            {"data": json.dumps(payload, default=str)},
            maxlen=settings.REDIS_STREAM_MAX_LEN,
            approximate=True,
        )
    except Exception:
        log.warning("redis_publish_failed", stream=settings.REDIS_LIVE_STREAM)
