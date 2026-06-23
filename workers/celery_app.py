"""Celery application configuration."""
import os

from celery import Celery

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

app = Celery(
    "vr180_studio",
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=["workers.convert_tasks", "workers.upscale_tasks"],
)

app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,  # GPU tasks — no prefetching
    result_expires=86400,  # Results kept for 24 hours
)
