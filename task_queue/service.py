from __future__ import annotations

from typing import Any, Dict

from celery.result import AsyncResult

from config import settings
from task_queue.celery_app import celery_app
from task_queue.tasks import (
    build_generation_task,
    queue_probe_task,
    run_batch_fusion_task,
    run_fusion_task,
    sync_vectors_task,
)


STATE_LABELS = {
    "PENDING": "queued",
    "RECEIVED": "received",
    "STARTED": "running",
    "PROGRESS": "running",
    "RETRY": "retrying",
    "SUCCESS": "succeeded",
    "FAILURE": "failed",
    "REVOKED": "cancelled",
}


def enqueue_probe(value: str = "ok") -> str:
    return queue_probe_task.apply_async(args=[value]).id


def enqueue_fusion(**kwargs: Any) -> str:
    return run_fusion_task.apply_async(kwargs=kwargs).id


def enqueue_batch_fusion(**kwargs: Any) -> str:
    return run_batch_fusion_task.apply_async(kwargs=kwargs).id


def enqueue_generation(*, name: str | None = None, publish: bool = True) -> str:
    return build_generation_task.apply_async(kwargs={"name": name, "publish": publish}).id


def enqueue_vector_sync(*, force: bool = False) -> str:
    return sync_vectors_task.apply_async(kwargs={"force": force}).id


def get_task_status(task_id: str) -> Dict[str, Any]:
    result: AsyncResult = celery_app.AsyncResult(task_id)
    state = str(result.state or "PENDING").upper()
    payload: Dict[str, Any] = {
        "task_id": task_id,
        "state": state,
        "status": STATE_LABELS.get(state, state.lower()),
        "ready": result.ready(),
        "successful": result.successful() if result.ready() else False,
    }
    info = result.info
    if state in {"STARTED", "PROGRESS", "RETRY"} and isinstance(info, dict):
        payload["progress"] = float(info.get("progress", 0))
        payload["stage"] = str(info.get("stage", ""))
    elif state == "SUCCESS":
        payload["progress"] = 1.0
        payload["result"] = result.result
    elif state == "FAILURE":
        payload["error"] = str(info)
    return payload


def revoke_task(task_id: str) -> Dict[str, Any]:
    # 不强杀正在执行的GPU进程，避免CUDA上下文和输出文件损坏。
    celery_app.control.revoke(task_id, terminate=False)
    return {"task_id": task_id, "cancel_requested": True}


def queue_status() -> Dict[str, Any]:
    try:
        import redis

        client = redis.Redis.from_url(
            settings.REDIS_HEALTH_URL,
            socket_connect_timeout=settings.REDIS_SOCKET_TIMEOUT_SECONDS,
            socket_timeout=settings.REDIS_SOCKET_TIMEOUT_SECONDS,
            decode_responses=True,
        )
        redis_ready = bool(client.ping())
        redis_version = str(client.info("server").get("redis_version", ""))
    except Exception as exc:
        return {
            "enabled": settings.TASK_QUEUE_ENABLED,
            "redis_ready": False,
            "workers": [],
            "error": str(exc),
        }

    replies = celery_app.control.inspect(timeout=0.8).ping() or {}
    return {
        "enabled": settings.TASK_QUEUE_ENABLED,
        "redis_ready": redis_ready,
        "redis_version": redis_version,
        "workers": sorted(replies.keys()),
        "worker_count": len(replies),
        "queues": [settings.CELERY_FUSION_QUEUE, settings.CELERY_MAINTENANCE_QUEUE],
        "result_expires_seconds": settings.CELERY_RESULT_EXPIRES_SECONDS,
    }
