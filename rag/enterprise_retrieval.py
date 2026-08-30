from __future__ import annotations

import hashlib
from typing import Any, Dict, Iterable, List, Optional

from config import settings
from rag import sqlite_kb
from rag.vector_store import vector_search


def _identity(hit: Dict[str, Any]) -> str:
    if hit.get("chunk_id") is not None:
        return f"chunk:{hit['chunk_id']}"
    raw = f"{hit.get('doc_title', '')}|{(hit.get('text') or hit.get('content') or '')[:300]}"
    return hashlib.sha1(raw.encode("utf-8", errors="ignore")).hexdigest()


def reciprocal_rank_fusion(
    ranked_lists: Iterable[List[Dict[str, Any]]],
    *,
    rrf_k: int = 60,
    weights: Optional[List[float]] = None,
) -> List[Dict[str, Any]]:
    """按排名而非原始分数融合关键词和向量结果。"""
    lists = list(ranked_lists)
    weights = weights or [1.0] * len(lists)
    if len(weights) != len(lists):
        raise ValueError("RRF 权重数量必须与召回列表数量一致")

    merged: Dict[str, Dict[str, Any]] = {}
    max_score = sum(weights) / float(rrf_k + 1)
    for route_index, (hits, weight) in enumerate(zip(lists, weights), start=1):
        for rank, hit in enumerate(hits, start=1):
            key = _identity(hit)
            item = merged.setdefault(key, dict(hit))
            for field, value in hit.items():
                if field not in item or item[field] in (None, "", []):
                    item[field] = value
            item.setdefault("retrieval_routes", [])
            item["retrieval_routes"].append(route_index)
            item["rrf_raw_score"] = item.get("rrf_raw_score", 0.0) + weight / (rrf_k + rank)

    for item in merged.values():
        item["rrf_score"] = round(item["rrf_raw_score"] / max_score, 6) if max_score else 0.0
        item["final_score"] = item["rrf_score"]
    return sorted(merged.values(), key=lambda item: item["rrf_raw_score"], reverse=True)


def expand_parent_context(
    child_sources: List[Dict[str, Any]], generation_id: Optional[int] = None
) -> List[Dict[str, Any]]:
    """根据子块 parent_id 回表，并合并命中同一父块的重复结果。"""
    parent_ids = [item.get("parent_id") for item in child_sources if item.get("parent_id")]
    parent_map = {
        item["parent_id"]: item
        for item in sqlite_kb.get_parent_chunks(parent_ids, generation_id=generation_id)
    }
    expanded: List[Dict[str, Any]] = []
    seen_parents: Dict[int, Dict[str, Any]] = {}
    for child in child_sources:
        parent_id = child.get("parent_id")
        parent = parent_map.get(parent_id)
        if not parent:
            expanded.append(dict(child))
            continue
        if parent_id in seen_parents:
            seen_parents[parent_id]["matched_child_ids"].append(child.get("chunk_id"))
            continue
        item = dict(child)
        item["matched_child_id"] = child.get("chunk_id")
        item["matched_child_text"] = child.get("text") or child.get("content") or ""
        item["matched_child_ids"] = [child.get("chunk_id")]
        item["parent_id"] = parent_id
        item["parent_index"] = parent.get("parent_index")
        item["title_path"] = parent.get("title_path", "")
        item["text"] = parent.get("content", "")
        item["content"] = parent.get("content", "")
        item["doc_title"] = parent.get("doc_title", item.get("doc_title", ""))
        item["category"] = parent.get("category", item.get("category", ""))
        item["source"] = parent.get("source", item.get("source", ""))
        seen_parents[parent_id] = item
        expanded.append(item)
    return expanded


def enterprise_search(
    query: str,
    top_k: int = 5,
    candidate_k: Optional[int] = None,
    reranker: Any = None,
    enable_reranker: Optional[bool] = None,
    generation_id: Optional[int] = None,
) -> Dict[str, Any]:
    """FTS/LIKE + Embedding 召回，经 RRF 融合和 BGE 精排。"""
    query = (query or "").strip()
    if not query:
        return {
            "query": query, "sources": [], "context": "", "source_text": "",
            "confidence": "low", "confidence_reason": "query 为空", "top_score": 0,
            "retrieval": {"sparse_count": 0, "vector_count": 0, "vector_used": False},
        }

    candidate_k = int(candidate_k or settings.RAG_CANDIDATE_K)
    rrf_k = int(settings.RAG_RRF_K)
    sparse_result = sqlite_kb.search_knowledge(
        query, top_k=candidate_k, generation_id=generation_id
    )
    sparse_hits = sparse_result.get("sources", []) or []
    vector_hits: List[Dict[str, Any]] = []
    warnings = []
    try:
        vector_hits = vector_search(
            query, top_k=candidate_k, generation_id=generation_id
        )
    except Exception as exc:
        warnings.append(f"向量检索暂不可用，已使用关键词检索：{exc}")

    fused = reciprocal_rank_fusion(
        [sparse_hits, vector_hits], rrf_k=rrf_k, weights=[1.0, 1.0]
    )
    reranker_enabled = settings.RERANKER_ENABLED if enable_reranker is None else enable_reranker
    reranker_used = False
    reranker_model = ""
    if reranker_enabled and fused:
        try:
            from rag.bge_reranker import rerank_candidates

            candidates = fused[:settings.RERANKER_CANDIDATE_K]
            child_sources = rerank_candidates(
                query,
                candidates,
                top_k=top_k,
                reranker=reranker,
            )
            reranker_used = True
            reranker_model = getattr(reranker, "model_name", "") or settings.RERANKER_MODEL
        except Exception as exc:
            warnings.append(f"BGE重排暂不可用，已保留RRF排序：{exc}")
            child_sources = fused[:top_k]
    else:
        child_sources = fused[:top_k]
    sources = expand_parent_context(child_sources, generation_id=generation_id)
    top_score = float(sources[0].get("final_score", 0)) if sources else 0.0
    if not sources:
        confidence, reason = "low", "未命中知识库内容"
    elif vector_hits and top_score >= 0.65:
        confidence, reason = "high", "关键词与语义向量检索提供了联合证据"
    elif top_score >= 0.35:
        confidence, reason = "medium", "已命中资料，但证据一致性一般"
    else:
        confidence, reason = "low", "检索依据较弱"

    context = "\n\n".join(
        f"【来源{i}】{item.get('doc_title', '未知文档')} | {item.get('category', '')}\n"
        f"{item.get('text') or item.get('content') or ''}"
        for i, item in enumerate(sources, start=1)
    )
    source_text = "\n".join(
        f"[{i}] {item.get('doc_title', '未知文档')} | RRF={item.get('final_score', 0):.4f}"
        for i, item in enumerate(sources, start=1)
    )
    return {
        "query": query,
        "sources": sources,
        "context": context,
        "source_text": source_text,
        "confidence": confidence,
        "confidence_reason": reason,
        "top_score": top_score,
        "retrieval": {
            "sparse_count": len(sparse_hits),
            "vector_count": len(vector_hits),
            "vector_used": bool(vector_hits),
            "fusion": "rrf",
            "rrf_k": rrf_k,
            "reranker_used": reranker_used,
            "reranker_model": reranker_model,
            "reranker_candidate_count": min(len(fused), settings.RERANKER_CANDIDATE_K),
            "child_result_count": len(child_sources),
            "parent_result_count": len(sources),
            "parent_lookup_used": any(item.get("parent_id") for item in sources),
            "generation_id": generation_id,
        },
        "warnings": warnings,
    }
