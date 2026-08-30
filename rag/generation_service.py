from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

from config import settings
from rag import sqlite_kb
from rag.vector_store import sync_chunk_embeddings


def _generation_name() -> str:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"gen-{stamp}-{uuid.uuid4().hex[:8]}"


def _event(conn, generation_id: int, event_type: str, detail: str = "") -> None:
    conn.execute(
        """
        INSERT INTO kb_generation_events(generation_id, event_type, detail, created_at)
        VALUES (?, ?, ?, ?)
        """,
        (generation_id, event_type, detail, sqlite_kb.now_str()),
    )


def create_generation(name: Optional[str] = None) -> Dict[str, Any]:
    sqlite_kb.init_db()
    generation_name = name or _generation_name()
    with sqlite_kb.connect() as conn:
        cursor = conn.execute(
            """
            INSERT INTO kb_generations(name, status, created_at, updated_at)
            VALUES (?, 'building', ?, ?)
            """,
            (generation_name, sqlite_kb.now_str(), sqlite_kb.now_str()),
        )
        generation_id = int(cursor.lastrowid)
        _event(conn, generation_id, "created", "开始构建草稿版本")
        conn.commit()
    return {"id": generation_id, "name": generation_name, "status": "building"}


def list_generations() -> List[Dict[str, Any]]:
    sqlite_kb.init_db()
    with sqlite_kb.connect() as conn:
        rows = conn.execute(
            """
            SELECT g.*,
                   COUNT(DISTINCT d.id) AS document_count,
                   COUNT(DISTINCT c.id) AS chunk_count,
                   COUNT(DISTINCT e.chunk_id) AS embedding_count,
                   ev.passed AS evaluation_passed,
                   ev.recall_at_k,
                   ev.mrr,
                   ev.ndcg_at_k,
                   aev.passed AS answer_evaluation_passed,
                   aev.answer_relevance,
                   aev.faithfulness,
                   aev.citation_completeness,
                   aev.refusal_correctness
            FROM kb_generations g
            LEFT JOIN documents d ON d.generation_id = g.id
            LEFT JOIN chunks c ON c.doc_id = d.id
            LEFT JOIN chunk_embeddings e ON e.chunk_id = c.id
            LEFT JOIN kb_generation_evaluations ev ON ev.generation_id = g.id
            LEFT JOIN kb_generation_answer_evaluations aev ON aev.generation_id = g.id
            GROUP BY g.id
            ORDER BY g.id DESC
            """
        ).fetchall()
    result = []
    for row in rows:
        item = dict(row)
        try:
            item["metadata"] = json.loads(item.pop("metadata_json") or "{}")
        except json.JSONDecodeError:
            item["metadata"] = {}
        result.append(item)
    return result


def generation_stats(generation_id: int) -> Dict[str, int]:
    sqlite_kb.init_db()
    with sqlite_kb.connect() as conn:
        row = conn.execute(
            """
            SELECT
                COUNT(DISTINCT d.id) AS document_count,
                COUNT(DISTINCT p.id) AS parent_count,
                COUNT(DISTINCT c.id) AS chunk_count,
                COUNT(DISTINCT CASE WHEN c.parent_id IS NULL THEN c.id END) AS unlinked_chunk_count,
                COUNT(DISTINCT e.chunk_id) AS embedding_count
            FROM documents d
            LEFT JOIN parent_chunks p ON p.doc_id = d.id
            LEFT JOIN chunks c ON c.doc_id = d.id
            LEFT JOIN chunk_embeddings e ON e.chunk_id = c.id
            WHERE d.generation_id = ?
            """,
            (generation_id,),
        ).fetchone()
    return {key: int(row[key] or 0) for key in row.keys()}


def validate_generation(generation_id: int, expected_documents: int) -> Dict[str, Any]:
    stats = generation_stats(generation_id)
    errors = []
    if stats["document_count"] != expected_documents:
        errors.append(
            f"文档数量不一致：expected={expected_documents}, actual={stats['document_count']}"
        )
    if stats["parent_count"] <= 0 or stats["chunk_count"] <= 0:
        errors.append("父块或子块为空")
    if stats["unlinked_chunk_count"]:
        errors.append(f"存在 {stats['unlinked_chunk_count']} 个未关联父块的子块")
    if stats["embedding_count"] != stats["chunk_count"]:
        errors.append(
            f"向量数量不一致：chunks={stats['chunk_count']}, embeddings={stats['embedding_count']}"
        )
    return {"valid": not errors, "errors": errors, **stats}


def _configured_files(folders: Optional[Iterable[Path]] = None) -> List[Path]:
    configured = list(folders or [settings.KNOWLEDGE_DIR, *settings.SUPPLEMENTAL_KNOWLEDGE_DIRS])
    files = []
    seen = set()
    for folder in configured:
        folder = Path(folder)
        if not folder.exists():
            continue
        for path in sorted(folder.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in sqlite_kb.SUPPORTED_EXTS:
                continue
            key = str(path.resolve()).lower()
            if key not in seen:
                seen.add(key)
                files.append(path.resolve())
    return files


def mark_generation_failed(generation_id: int, error: str) -> None:
    with sqlite_kb.connect() as conn:
        conn.execute(
            """
            UPDATE kb_generations
            SET status = 'failed', error_message = ?, updated_at = ?
            WHERE id = ? AND status IN ('building', 'ready')
            """,
            (error[:4000], sqlite_kb.now_str(), generation_id),
        )
        _event(conn, generation_id, "failed", error[:1000])
        conn.commit()


def check_evaluation_gate(
    report: Dict[str, Any],
    thresholds: Optional[Dict[str, float]] = None,
    min_questions: Optional[int] = None,
) -> Dict[str, Any]:
    thresholds = dict(thresholds or settings.RAG_EVAL_THRESHOLDS)
    min_questions = int(
        settings.RAG_EVAL_MIN_QUESTIONS if min_questions is None else min_questions
    )
    failures = []
    total = int(report.get("total", 0))
    if total < min_questions:
        failures.append(f"评测题不足：{total} < {min_questions}")
    for metric, threshold in thresholds.items():
        actual = float(report.get(metric, 0.0))
        if actual < float(threshold):
            failures.append(f"{metric}={actual:.4f} < {float(threshold):.4f}")
    return {
        "passed": not failures,
        "failures": failures,
        "thresholds": thresholds,
        "min_questions": min_questions,
    }


def get_generation_evaluation(generation_id: int) -> Dict[str, Any]:
    sqlite_kb.init_db()
    with sqlite_kb.connect() as conn:
        row = conn.execute(
            "SELECT * FROM kb_generation_evaluations WHERE generation_id = ?",
            (generation_id,),
        ).fetchone()
    if row is None:
        return {}
    result = dict(row)
    for source, target in (("thresholds_json", "thresholds"), ("report_json", "report")):
        try:
            result[target] = json.loads(result.pop(source) or "{}")
        except json.JSONDecodeError:
            result[target] = {}
    result["passed"] = bool(result["passed"])
    return result


def get_generation_answer_evaluation(generation_id: int) -> Dict[str, Any]:
    sqlite_kb.init_db()
    with sqlite_kb.connect() as conn:
        row = conn.execute(
            "SELECT * FROM kb_generation_answer_evaluations WHERE generation_id = ?",
            (generation_id,),
        ).fetchone()
    if row is None:
        return {}
    result = dict(row)
    for source, target in (("thresholds_json", "thresholds"), ("report_json", "report")):
        try:
            result[target] = json.loads(result.pop(source) or "{}")
        except json.JSONDecodeError:
            result[target] = {}
    result["passed"] = bool(result["passed"])
    return result


def evaluate_generation(
    generation_id: int,
    *,
    eval_items: Optional[List[Dict[str, Any]]] = None,
    eval_path: Optional[Path] = None,
    thresholds: Optional[Dict[str, float]] = None,
    min_questions: Optional[int] = None,
    search_fn: Optional[Callable[[str], Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    from rag.enterprise_retrieval import enterprise_search
    from rag.eval_service import evaluate_retrieval, load_eval_set

    path = Path(eval_path or settings.RAG_EVAL_SET_PATH)
    items = list(eval_items) if eval_items is not None else load_eval_set(path)
    if search_fn is None:
        search_fn = lambda question: enterprise_search(
            question,
            top_k=settings.RAG_EVAL_TOP_K,
            generation_id=generation_id,
        )
    report = evaluate_retrieval(items, search_fn, top_k=settings.RAG_EVAL_TOP_K)
    gate = check_evaluation_gate(report, thresholds, min_questions)
    report["gate"] = gate
    with sqlite_kb.connect() as conn:
        conn.execute(
            """
            INSERT INTO kb_generation_evaluations(
                generation_id, eval_set_path, total, recall_at_k, mrr, ndcg_at_k,
                top1_accuracy, keyword_coverage, passed, thresholds_json,
                report_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(generation_id) DO UPDATE SET
                eval_set_path=excluded.eval_set_path, total=excluded.total,
                recall_at_k=excluded.recall_at_k, mrr=excluded.mrr,
                ndcg_at_k=excluded.ndcg_at_k,
                top1_accuracy=excluded.top1_accuracy,
                keyword_coverage=excluded.keyword_coverage, passed=excluded.passed,
                thresholds_json=excluded.thresholds_json,
                report_json=excluded.report_json, created_at=excluded.created_at
            """,
            (
                generation_id, str(path), report["total"], report["recall_at_k"],
                report["mrr"], report["ndcg_at_k"], report["top1_accuracy"],
                report["keyword_coverage"], int(gate["passed"]),
                json.dumps(gate, ensure_ascii=False),
                json.dumps(report, ensure_ascii=False), sqlite_kb.now_str(),
            ),
        )
        _event(conn, generation_id, "evaluation_passed" if gate["passed"] else "evaluation_failed", json.dumps(gate, ensure_ascii=False))
        conn.commit()
    return report


def evaluate_generation_answers(
    generation_id: int,
    *,
    eval_items: Optional[List[Dict[str, Any]]] = None,
    eval_path: Optional[Path] = None,
    thresholds: Optional[Dict[str, float]] = None,
    min_questions: Optional[int] = None,
    search_fn: Optional[Callable[[str], Dict[str, Any]]] = None,
    answer_fn: Optional[Callable[[str, List[Dict[str, Any]]], str]] = None,
    judge_fn: Optional[Callable[[Dict[str, Any], str, List[Dict[str, Any]]], Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    from rag.answer_eval_service import (
        check_answer_quality_gate,
        evaluate_answer_quality,
        load_answer_eval_set,
    )
    from rag.enterprise_retrieval import enterprise_search

    path = Path(eval_path or settings.RAG_ANSWER_EVAL_SET_PATH)
    items = list(eval_items) if eval_items is not None else load_answer_eval_set(path)
    if search_fn is None:
        search_fn = lambda question: enterprise_search(
            question,
            top_k=settings.RAG_EVAL_TOP_K,
            generation_id=generation_id,
        )
    report = evaluate_answer_quality(
        items,
        search_fn,
        answer_fn=answer_fn,
        judge_fn=judge_fn,
    )
    gate = check_answer_quality_gate(report, thresholds, min_questions)
    report["gate"] = gate
    with sqlite_kb.connect() as conn:
        conn.execute(
            """
            INSERT INTO kb_generation_answer_evaluations(
                generation_id, eval_set_path, total, answer_relevance,
                faithfulness, citation_completeness, refusal_correctness,
                passed, thresholds_json, report_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(generation_id) DO UPDATE SET
                eval_set_path=excluded.eval_set_path, total=excluded.total,
                answer_relevance=excluded.answer_relevance,
                faithfulness=excluded.faithfulness,
                citation_completeness=excluded.citation_completeness,
                refusal_correctness=excluded.refusal_correctness,
                passed=excluded.passed, thresholds_json=excluded.thresholds_json,
                report_json=excluded.report_json, created_at=excluded.created_at
            """,
            (
                generation_id, str(path), report["total"],
                report["answer_relevance"], report["faithfulness"],
                report["citation_completeness"], report["refusal_correctness"],
                int(gate["passed"]), json.dumps(gate, ensure_ascii=False),
                json.dumps(report, ensure_ascii=False), sqlite_kb.now_str(),
            ),
        )
        _event(
            conn,
            generation_id,
            "answer_evaluation_passed" if gate["passed"] else "answer_evaluation_failed",
            json.dumps(gate, ensure_ascii=False),
        )
        conn.commit()
    return report


def build_generation(
    *,
    name: Optional[str] = None,
    folders: Optional[Iterable[Path]] = None,
    embedder: Any = None,
    publish: bool = False,
    evaluation_enabled: Optional[bool] = None,
    eval_items: Optional[List[Dict[str, Any]]] = None,
    thresholds: Optional[Dict[str, float]] = None,
    min_questions: Optional[int] = None,
    answer_evaluation_enabled: Optional[bool] = None,
    answer_eval_items: Optional[List[Dict[str, Any]]] = None,
    answer_thresholds: Optional[Dict[str, float]] = None,
    answer_min_questions: Optional[int] = None,
) -> Dict[str, Any]:
    generation = create_generation(name)
    generation_id = generation["id"]
    files = _configured_files(folders)
    try:
        if not files:
            raise RuntimeError("没有找到可入库文档")
        imported = []
        for path in files:
            doc_id = sqlite_kb.add_file_document(
                path,
                title=path.name,
                category=sqlite_kb.infer_category(path.name),
                generation_id=generation_id,
            )
            imported.append({"doc_id": doc_id, "file": str(path)})

        vector_result = sync_chunk_embeddings(
            generation_id=generation_id,
            embedder=embedder,
        )
        if not vector_result.get("ready"):
            raise RuntimeError(
                "向量后端未就绪："
                f"indexed={vector_result.get('indexed_chunks', 0)}, "
                f"total={vector_result.get('total_chunks', 0)}"
            )
        validation = validate_generation(generation_id, expected_documents=len(files))
        if not validation["valid"]:
            raise RuntimeError("；".join(validation["errors"]))

        should_evaluate = (
            settings.RAG_EVALUATION_ENABLED
            if evaluation_enabled is None else evaluation_enabled
        )
        evaluation = None
        answer_evaluation = None
        if should_evaluate:
            evaluation = evaluate_generation(
                generation_id,
                eval_items=eval_items,
                thresholds=thresholds,
                min_questions=min_questions,
            )
            if not evaluation["gate"]["passed"]:
                raise RuntimeError(
                    "RAG评测门禁未通过：" + "；".join(evaluation["gate"]["failures"])
                )
            should_evaluate_answers = (
                settings.RAG_ANSWER_EVALUATION_ENABLED
                if answer_evaluation_enabled is None else answer_evaluation_enabled
            )
            if should_evaluate_answers:
                answer_evaluation = evaluate_generation_answers(
                    generation_id,
                    eval_items=answer_eval_items,
                    thresholds=answer_thresholds,
                    min_questions=answer_min_questions,
                )
                if not answer_evaluation["gate"]["passed"]:
                    raise RuntimeError(
                        "回答质量门禁未通过："
                        + "；".join(answer_evaluation["gate"]["failures"])
                    )

        metadata = {
            "files": [str(path) for path in files],
            "embedding_model": vector_result.get("embedding_model"),
            "validation": validation,
            "evaluation": evaluation,
            "answer_evaluation": answer_evaluation,
        }
        with sqlite_kb.connect() as conn:
            conn.execute(
                """
                UPDATE kb_generations
                SET status = 'ready', metadata_json = ?, updated_at = ?, error_message = ''
                WHERE id = ? AND status = 'building'
                """,
                (json.dumps(metadata, ensure_ascii=False), sqlite_kb.now_str(), generation_id),
            )
            _event(conn, generation_id, "validated", json.dumps(validation, ensure_ascii=False))
            conn.commit()

        publish_result = publish_generation(generation_id) if publish else None
        return {
            "generation_id": generation_id,
            "name": generation["name"],
            "status": "active" if publish else "ready",
            "imported": imported,
            "vector_index": vector_result,
            "validation": validation,
            "evaluation": evaluation,
            "answer_evaluation": answer_evaluation,
            "publish": publish_result,
        }
    except Exception as exc:
        mark_generation_failed(generation_id, str(exc))
        raise


def publish_generation(
    generation_id: int,
    *,
    allow_retired: bool = False,
    event_type: str = "published",
    require_gate: bool = True,
    _before_activate: Optional[Callable[[], None]] = None,
) -> Dict[str, Any]:
    """单事务切换 active generation；任一步失败都会恢复旧版本。"""
    sqlite_kb.init_db()
    allowed = ("ready", "retired") if allow_retired else ("ready",)
    placeholders = ",".join("?" for _ in allowed)
    with sqlite_kb.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        target = conn.execute(
            f"SELECT id, name, status FROM kb_generations WHERE id = ? AND status IN ({placeholders})",
            (generation_id, *allowed),
        ).fetchone()
        if target is None:
            raise ValueError("目标 generation 不存在或状态不允许发布")
        if require_gate and not allow_retired:
            evaluation = conn.execute(
                "SELECT passed FROM kb_generation_evaluations WHERE generation_id = ?",
                (generation_id,),
            ).fetchone()
            if evaluation is None or not bool(evaluation["passed"]):
                raise ValueError("目标 generation 未通过RAG评测门禁，禁止发布")
            if settings.RAG_ANSWER_EVALUATION_ENABLED:
                answer_evaluation = conn.execute(
                    "SELECT passed FROM kb_generation_answer_evaluations WHERE generation_id = ?",
                    (generation_id,),
                ).fetchone()
                if answer_evaluation is None or not bool(answer_evaluation["passed"]):
                    raise ValueError("目标 generation 未通过回答质量门禁，禁止发布")
        previous = conn.execute(
            "SELECT id, name FROM kb_generations WHERE status = 'active' LIMIT 1"
        ).fetchone()
        if previous and int(previous["id"]) != generation_id:
            conn.execute(
                "UPDATE kb_generations SET status = 'retired', updated_at = ? WHERE id = ?",
                (sqlite_kb.now_str(), int(previous["id"])),
            )
        if _before_activate:
            _before_activate()
        conn.execute(
            """
            UPDATE kb_generations
            SET status = 'active', published_at = ?, updated_at = ?, error_message = ''
            WHERE id = ?
            """,
            (sqlite_kb.now_str(), sqlite_kb.now_str(), generation_id),
        )
        _event(
            conn,
            generation_id,
            event_type,
            f"previous_generation_id={int(previous['id']) if previous else ''}",
        )
        conn.commit()
    return {
        "active_generation_id": generation_id,
        "active_generation": target["name"],
        "previous_generation_id": int(previous["id"]) if previous else None,
    }


def rollback_generation(generation_id: int) -> Dict[str, Any]:
    return publish_generation(
        generation_id,
        allow_retired=True,
        event_type="rolled_back",
        require_gate=False,
    )
