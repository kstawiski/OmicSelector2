"""Celery task definitions for asynchronous job execution.

Background tasks for computationally intensive operations:
- Feature selection jobs
- Model training jobs
- Preprocessing pipelines
- Benchmarking tasks
- Data quality control
"""

try:
    from celery import Celery

    CELERY_AVAILABLE = True
except ImportError:
    CELERY_AVAILABLE = False
    Celery = None  # type: ignore

from omicselector2.utils.config import get_settings


def create_celery_app() -> "Celery":  # type: ignore
    """Create and configure Celery application.

    Returns:
        Configured Celery application instance

    Raises:
        ImportError: If Celery is not installed
    """
    if not CELERY_AVAILABLE:
        raise ImportError(
            "Celery is required for task queue functionality. "
            "Install with: pip install celery redis"
        )

    settings = get_settings()

    celery_app = Celery(
        "omicselector2",
        broker=settings.CELERY_BROKER_URL,
        backend=settings.CELERY_RESULT_BACKEND,
    )

    # Configure Celery
    celery_app.conf.update(
        task_serializer="json",
        accept_content=["json"],
        result_serializer="json",
        timezone="UTC",
        enable_utc=True,
        task_track_started=True,
        task_time_limit=3600,  # 1 hour hard limit
        task_soft_time_limit=3300,  # 55 minutes soft limit
        worker_prefetch_multiplier=1,
        worker_max_tasks_per_child=1000,
        task_acks_late=True,
        task_reject_on_worker_lost=True,
        result_expires=86400,  # 24 hours
    )

    return celery_app


# Create Celery app instance
if CELERY_AVAILABLE:
    celery_app = create_celery_app()
else:
    celery_app = None  # type: ignore


__all__ = ["celery_app", "create_celery_app"]
