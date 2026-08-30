from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from celery.exceptions import SoftTimeLimitExceeded

from config import settings
from task_queue.celery_app import celery_app


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return str(value)


def _safe_uploaded_path(path_value: str) -> Path:
    path = Path(path_value).resolve()
    temp_root = Path(settings.TEMP_DIR).resolve()
    try:
        path.relative_to(temp_root)
    except ValueError as exc:
        raise ValueError("异步融合输入只能来自项目上传临时目录") from exc
    if not path.is_file():
        raise FileNotFoundError(f"找不到融合输入文件：{path}")
    return path


def _output_url(path_value: Any) -> str | None:
    if not path_value:
        return None


def _enrich_output_urls(result: Dict[str, Any]) -> Dict[str, Any]:
    safe_result = _json_safe(result)
    safe_result["urls"] = {
        key.replace("_path", "_url"): _output_url(value)
        for key, value in safe_result.items()
        if key.endswith("_path") and value
    }
    return safe_result


def _batch_report(results: list[Dict[str, Any]]) -> str:
    lines = [
        "# MambaDFuse-Agent 批量红外—可见光融合报告",
        "",
        "| 序号 | 可见光 | 红外 | 综合评分 | Qabf | MI | SSIM |",
        "|---:|---|---|---:|---:|---:|---:|",
    ]
    for item in results:
        metrics = item.get("metrics", {}) or {}
        lines.append(
            "| {index} | {optical} | {infrared} | {score:.2f} | {qabf:.4f} | {mi:.4f} | {ssim:.4f} |".format(
                index=item.get("_pair_index", ""),
                optical=item.get("_optical_name", ""),
                infrared=item.get("_sar_name", ""),
                score=float(metrics.get("综合质量评分", 0) or 0),
                qabf=float(metrics.get("Qabf 边缘保持", 0) or 0),
                mi=float(metrics.get("MI 互信息", 0) or 0),
                ssim=float(metrics.get("SSIM 结构相似性", 0) or 0),
            )
        )
    return "\n".join(lines)
    try:
        relative = Path(str(path_value)).resolve().relative_to(Path(settings.OUTPUT_DIR).resolve())
        return "/outputs/" + relative.as_posix()
    except Exception:
        return None


@celery_app.task(bind=True, name="mambadfuse.queue.probe")
def queue_probe_task(self, value: str = "ok") -> Dict[str, Any]:
    self.update_state(state="PROGRESS", meta={"progress": 0.5, "stage": "worker已接收任务"})
    return {"task_type": "probe", "worker_ready": True, "value": value}


@celery_app.task(bind=True, name="mambadfuse.fusion.run")
def run_fusion_task(
    self,
    optical_path: str,
    infrared_path: str,
    use_tile: bool = False,
    tile: int = 256,
    overlap: int = 32,
    generate_report: bool = False,
    user_note: str = "",
) -> Dict[str, Any]:
    optical: Path | None = None
    infrared: Path | None = None
    try:
        optical = _safe_uploaded_path(optical_path)
        infrared = _safe_uploaded_path(infrared_path)
        self.update_state(state="PROGRESS", meta={"progress": 0.1, "stage": "校验输入完成"})

        from fusion.wemfusion_runner import run_wemfusion

        self.update_state(state="PROGRESS", meta={"progress": 0.25, "stage": "加载MambaDFuse并开始推理"})
        result = run_wemfusion(
            optical_path=str(optical),
            sar_path=str(infrared),
            output_dir=str(settings.OUTPUT_DIR),
            use_tile=bool(use_tile),
            tile=int(tile),
            overlap=int(overlap),
        )
        self.update_state(state="PROGRESS", meta={"progress": 0.9, "stage": "融合完成，整理结果"})
        result = _enrich_output_urls(result)
        report = None
        if generate_report:
            from reports.report_generator import generate_report

            report = generate_report(result, user_note=user_note)
        return {
            "task_type": "fusion",
            "fusion_result": result,
            "report_text": report,
        }
    except SoftTimeLimitExceeded as exc:
        raise TimeoutError("MambaDFuse任务超过软超时限制") from exc
    finally:
        # 任务参数只保存路径，上传源文件在成功或失败后都应清理；融合输出单独保留。
        for source in (optical, infrared):
            if source is not None:
                source.unlink(missing_ok=True)


@celery_app.task(bind=True, name="mambadfuse.fusion.batch")
def run_batch_fusion_task(
    self,
    optical_items: list[Dict[str, str]],
    infrared_items: list[Dict[str, str]],
    use_tile: bool = False,
    tile: int = 256,
    overlap: int = 32,
    generate_report: bool = False,
) -> Dict[str, Any]:
    all_paths = [item.get("path", "") for item in optical_items + infrared_items]
    try:
        pair_count = min(len(optical_items), len(infrared_items))
        if pair_count <= 0:
            raise ValueError("没有可配对的图像")

        from fusion.wemfusion_runner import run_wemfusion

        results: list[Dict[str, Any]] = []
        for index in range(pair_count):
            optical = _safe_uploaded_path(optical_items[index]["path"])
            infrared = _safe_uploaded_path(infrared_items[index]["path"])
            self.update_state(
                state="PROGRESS",
                meta={
                    "progress": index / pair_count,
                    "stage": f"正在处理第 {index + 1}/{pair_count} 组",
                },
            )
            result = run_wemfusion(
                optical_path=str(optical),
                sar_path=str(infrared),
                output_dir=str(settings.OUTPUT_DIR),
                use_tile=bool(use_tile),
                tile=int(tile),
                overlap=int(overlap),
            )
            result = _enrich_output_urls(result)
            result["_pair_index"] = index + 1
            result["_optical_name"] = optical_items[index].get("name", optical.name)
            result["_sar_name"] = infrared_items[index].get("name", infrared.name)
            results.append(result)

        return {
            "task_type": "batch_fusion",
            "answer": f"批量融合完成：成功处理 {len(results)}/{pair_count} 组图像。",
            "batch_results": results,
            "report_text": _batch_report(results) if generate_report else None,
        }
    except SoftTimeLimitExceeded as exc:
        raise TimeoutError("批量MambaDFuse任务超过软超时限制") from exc
    finally:
        for path_value in all_paths:
            try:
                _safe_uploaded_path(path_value).unlink(missing_ok=True)
            except (ValueError, FileNotFoundError):
                pass


@celery_app.task(
    bind=True,
    name="mambadfuse.rag.generation",
    autoretry_for=(ConnectionError, TimeoutError),
    retry_backoff=True,
    retry_backoff_max=settings.CELERY_RETRY_BACKOFF_MAX_SECONDS,
    retry_jitter=True,
    max_retries=settings.CELERY_MAX_RETRIES,
)
def build_generation_task(
    self,
    name: str | None = None,
    publish: bool = True,
) -> Dict[str, Any]:
    try:
        self.update_state(state="PROGRESS", meta={"progress": 0.05, "stage": "扫描知识文档"})
        from rag.generation_service import build_generation

        result = build_generation(name=name, publish=bool(publish))
        self.update_state(state="PROGRESS", meta={"progress": 0.95, "stage": "知识库版本构建完成"})
        return {"task_type": "rag_generation", **_json_safe(result)}
    except SoftTimeLimitExceeded as exc:
        raise TimeoutError("知识库构建任务超过软超时限制") from exc


@celery_app.task(
    bind=True,
    name="mambadfuse.rag.vector_sync",
    autoretry_for=(ConnectionError, TimeoutError),
    retry_backoff=True,
    retry_backoff_max=settings.CELERY_RETRY_BACKOFF_MAX_SECONDS,
    retry_jitter=True,
    max_retries=settings.CELERY_MAX_RETRIES,
)
def sync_vectors_task(self, force: bool = False) -> Dict[str, Any]:
    try:
        self.update_state(state="PROGRESS", meta={"progress": 0.1, "stage": "准备向量同步"})
        from rag.vector_store import sync_chunk_embeddings

        result = sync_chunk_embeddings(force=bool(force))
        self.update_state(state="PROGRESS", meta={"progress": 0.95, "stage": "向量同步完成"})
        return {"task_type": "vector_sync", **_json_safe(result)}
    except SoftTimeLimitExceeded as exc:
        raise TimeoutError("向量同步任务超过软超时限制") from exc
