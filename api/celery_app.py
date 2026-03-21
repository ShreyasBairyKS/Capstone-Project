"""
api/celery_app.py — Celery worker configuration for VisionFood QAI.

Provides the Celery application instance and the PDF generation task.
The broker and result backend both use Redis.
"""

from __future__ import annotations

from celery import Celery

from core.config import settings

celery_app = Celery(
    "visionfood_qai",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
    include=["reports.tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    # Prevent piling up stale results
    result_expires=86400,  # 24 hours
    # Retry settings
    task_acks_late=True,
    task_reject_on_worker_lost=True,
)
