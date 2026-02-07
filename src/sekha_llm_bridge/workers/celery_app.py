"""Celery configuration for background jobs"""

from celery import Celery

from sekha_llm_bridge.config import get_settings

_celery_app = None


def get_celery_app() -> Celery:
    """Get or create the Celery app instance (lazy initialization)."""
    global _celery_app

    if _celery_app is not None:
        return _celery_app

    # Get settings (will raise if not loaded)
    settings = get_settings()

    _celery_app = Celery(
        "sekha_llm_bridge",
        broker=settings.celery_broker_url,
        backend=settings.celery_result_backend,
        include=["workers.tasks"],
    )

    # Configuration
    _celery_app.conf.update(
        task_serializer="json",
        accept_content=["json"],
        result_serializer="json",
        timezone="UTC",
        enable_utc=True,
        task_track_started=True,
        task_time_limit=300,  # 5 minutes
        worker_prefetch_multiplier=1,
    )

    return _celery_app


# For backwards compatibility, create a proxy that lazily initializes
class _CeleryAppProxy:
    def __getattr__(self, name):
        return getattr(get_celery_app(), name)


celery_app = _CeleryAppProxy()
