"""
api/routers/websocket.py — Live inspection stream via WebSocket.

Route:
  WS  /ws/live   — Streams live inspection events to connected clients.

Events are published to the Redis Stream `inspections:live` by
core.messaging.publish_inspection_event() after every inspection.
This router reads that stream and fans out to all connected WebSocket clients.
"""

from __future__ import annotations

import asyncio
import json

import redis.asyncio as aioredis
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from core.config import settings
from core.logging import get_logger

log = get_logger(__name__)
router = APIRouter(tags=["WebSocket"])

# Polling interval for the Redis stream (seconds)
_POLL_INTERVAL = 0.25
# How many events to serve per poll to avoid blocking
_READ_COUNT = 10


@router.websocket("/ws/live")
async def live_inspection_stream(websocket: WebSocket) -> None:
    """
    WebSocket endpoint that pushes inspection events in real-time.

    The client connects and receives JSON-encoded InspectionResult payloads
    as they arrive from the Redis Stream. No authentication is enforced at
    the WebSocket level — deploy behind a TLS-terminating reverse proxy.
    """
    await websocket.accept()
    log.info("ws_client_connected", client=str(websocket.client))

    try:
        r = aioredis.from_url(settings.REDIS_URL, decode_responses=True)
        # Start reading from the tip of the stream ($) so we only see new events
        last_id = "$"

        while True:
            try:
                entries = await r.xread(
                    {settings.REDIS_LIVE_STREAM: last_id},
                    count=_READ_COUNT,
                    block=int(_POLL_INTERVAL * 1000),
                )
            except (aioredis.ConnectionError, aioredis.TimeoutError) as exc:
                log.warning("ws_redis_error", error=str(exc))
                await asyncio.sleep(1)
                continue

            if entries:
                stream_name, messages = entries[0]
                for msg_id, fields in messages:
                    last_id = msg_id
                    try:
                        await websocket.send_text(json.dumps(fields))
                    except WebSocketDisconnect:
                        raise

        await r.aclose()

    except WebSocketDisconnect:
        log.info("ws_client_disconnected", client=str(websocket.client))
    except Exception as exc:
        log.error("ws_unexpected_error", error=str(exc))
        try:
            await websocket.close(code=1011)
        except Exception:
            pass
