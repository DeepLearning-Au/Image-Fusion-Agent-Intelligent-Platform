from __future__ import annotations

from task_queue.celery_app import celery_app
from task_queue import service


class _FakeResult:
    def __init__(self, state: str, info=None, result=None):
        self.state = state
        self.info = info
        self.result = result

    def ready(self) -> bool:
        return self.state in {"SUCCESS", "FAILURE", "REVOKED"}

    def successful(self) -> bool:
        return self.state == "SUCCESS"


def test_celery_routes_long_tasks_to_dedicated_queues():
    routes = celery_app.conf.task_routes
    assert routes["mambadfuse.fusion.run"]["queue"] == "gpu"
    assert routes["mambadfuse.fusion.batch"]["queue"] == "gpu"
    assert routes["mambadfuse.rag.generation"]["queue"] == "maintenance"
    assert routes["mambadfuse.rag.vector_sync"]["queue"] == "maintenance"
    assert celery_app.conf.worker_prefetch_multiplier == 1


def test_task_progress_is_normalized(monkeypatch):
    monkeypatch.setattr(
        service.celery_app,
        "AsyncResult",
        lambda _task_id: _FakeResult(
            "PROGRESS",
            info={"progress": 0.35, "stage": "正在执行"},
        ),
    )
    status = service.get_task_status("task-1")
    assert status["status"] == "running"
    assert status["progress"] == 0.35
    assert status["stage"] == "正在执行"


def test_task_success_returns_json_result(monkeypatch):
    expected = {"task_type": "probe", "worker_ready": True}
    monkeypatch.setattr(
        service.celery_app,
        "AsyncResult",
        lambda _task_id: _FakeResult("SUCCESS", result=expected),
    )
    status = service.get_task_status("task-2")
    assert status["status"] == "succeeded"
    assert status["progress"] == 1.0
    assert status["result"] == expected
