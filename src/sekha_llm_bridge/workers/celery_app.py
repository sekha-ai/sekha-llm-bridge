"""Celery configuration for background jobs"""

from functools import wraps
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


class _LazyTaskDecorator:
    """Lazy task decorator that delays registration until first use."""

    def __init__(self, **task_kwargs):
        self.task_kwargs = task_kwargs
        self._registered_task = None

    def __call__(self, func):
        """Decorate a function as a Celery task."""

        @wraps(func)
        def wrapper(*args, **kwargs):
            # Register task on first actual call
            if self._registered_task is None:
                app = get_celery_app()
                self._registered_task = app.task(**self.task_kwargs)(func)
            return self._registered_task(*args, **kwargs)

        # Add task-like attributes for Celery
        def _get_registered():
            if wrapper._registered_task is None:
                app = get_celery_app()
                wrapper._registered_task = app.task(**self.task_kwargs)(func)
            return wrapper._registered_task

        wrapper._registered_task = None
        wrapper.delay = lambda *args, **kwargs: _get_registered().delay(*args, **kwargs)
        wrapper.apply_async = lambda *args, **kwargs: _get_registered().apply_async(
            *args, **kwargs
        )
        wrapper.s = lambda *args, **kwargs: _get_registered().s(*args, **kwargs)
        wrapper.signature = lambda *args, **kwargs: _get_registered().signature(
            *args, **kwargs
        )

        return wrapper


# For backwards compatibility, create a proxy that lazily initializes
class _CeleryAppProxy:
    """Proxy for celery_app that delays initialization."""

    def task(self, *args, **kwargs):
        """Return a lazy task decorator."""
        # If called as @celery_app.task (without parentheses)
        if args and callable(args[0]):
            return _LazyTaskDecorator()(args[0])
        else:
            # Called with arguments: @celery_app.task(name="...")
            return _LazyTaskDecorator(**kwargs)

    def __getattr__(self, name):
        """Forward other attributes to the real celery app."""
        if name == "task":
            return self.task
        return getattr(get_celery_app(), name)


celery_app = _CeleryAppProxy()
