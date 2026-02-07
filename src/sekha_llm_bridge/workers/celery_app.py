"""Celery configuration for background jobs"""

import os

from celery import Celery

from sekha_llm_bridge.config import settings

# Use environment variables for Celery broker/backend since they're infrastructure config
# not model/routing config that goes in the YAML
celery_app = Celery(
    "sekha_llm_bridge",
    broker=os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0"),
    backend=os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/0"),
    include=["workers.tasks"],
)

# Configuration
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=300,  # 5 minutes
    worker_prefetch_multiplier=1,
)
