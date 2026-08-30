"""Redis/Celery异步任务基础设施。"""

from task_queue.celery_app import celery_app

__all__ = ["celery_app"]
