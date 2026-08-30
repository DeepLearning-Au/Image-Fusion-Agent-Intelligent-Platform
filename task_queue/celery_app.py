from __future__ import annotations

from celery import Celery

from config import settings


celery_app = Celery(
    "mambadfuse_agent",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    include=["task_queue.tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="Asia/Shanghai",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,
    task_acks_on_failure_or_timeout=True,
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=settings.CELERY_WORKER_PREFETCH_MULTIPLIER,
    worker_cancel_long_running_tasks_on_connection_loss=True,
    broker_connection_retry_on_startup=True,
    broker_transport_options={
        "visibility_timeout": settings.CELERY_VISIBILITY_TIMEOUT_SECONDS,
    },
    result_backend_transport_options={
        "visibility_timeout": settings.CELERY_VISIBILITY_TIMEOUT_SECONDS,
        "global_keyprefix": "mambadfuse:",
    },
    result_expires=settings.CELERY_RESULT_EXPIRES_SECONDS,
    task_soft_time_limit=settings.CELERY_SOFT_TIME_LIMIT_SECONDS,
    task_time_limit=settings.CELERY_HARD_TIME_LIMIT_SECONDS,
    task_send_sent_event=True,
    worker_send_task_events=True,
    task_default_queue=settings.CELERY_MAINTENANCE_QUEUE,
    task_routes={
        "mambadfuse.fusion.run": {"queue": settings.CELERY_FUSION_QUEUE},
        "mambadfuse.fusion.batch": {"queue": settings.CELERY_FUSION_QUEUE},
        "mambadfuse.rag.generation": {"queue": settings.CELERY_MAINTENANCE_QUEUE},
        "mambadfuse.rag.vector_sync": {"queue": settings.CELERY_MAINTENANCE_QUEUE},
        "mambadfuse.queue.probe": {"queue": settings.CELERY_MAINTENANCE_QUEUE},
    },
)
