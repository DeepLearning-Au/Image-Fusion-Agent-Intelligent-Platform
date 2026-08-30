import re
from typing import Any, Dict, List


def _sqlite_results(query: str, top_k: int) -> List[Dict[str, Any]]:
    """查询企业知识库：关键词与向量双路召回后使用 RRF 融合。"""
    try:
        from rag.enterprise_retrieval import enterprise_search
        result = enterprise_search(query, top_k=top_k)
        return result.get("sources", []) or []
    except Exception:
        return []


def _vector_results(query: str, top_k: int) -> List[Dict[str, Any]]:
    """保留旧向量索引作为兼容回退，避免数据库尚未重建时知识问答完全不可用。"""
    try:
        from rag.vector_store import retrieve_from_vector_store
        return retrieve_from_vector_store(query, top_k=top_k) or []
    except Exception:
        return []


def rag_retrieve(query: str, top_k: int = 5) -> str:
    """
    RAG 检索工具。
    返回给 Agent 使用。
    """
    results = _sqlite_results(query, top_k=top_k)
    source_mode = "SQLite企业知识库"

    if not results:
        results = _vector_results(query, top_k=top_k)
        source_mode = "旧向量索引回退"

    if not results:
        return "知识库中没有检索到相关内容。"

    output = []

    for index, item in enumerate(results, start=1):
        title = item.get("doc_title") or item.get("title") or item.get("source") or "未知来源"
        source = item.get("source", "")
        category = item.get("category", "")
        score = item.get("final_score", item.get("score", 0))
        content = item.get("text") or item.get("content") or ""
        page_match = re.search(r"\[PDF_PAGE=(\d+)\]", content)
        page = page_match.group(1) if page_match else item.get("page", "")
        output.append(
            f"【资料{index}】\n"
            f"检索方式：{source_mode}\n"
            f"文档：{title}\n"
            f"类别：{category}\n"
            f"来源：{source}\n"
            f"页码：{page or '未标注'}\n"
            f"相关度：{float(score or 0):.4f}\n"
            f"内容：\n{content[:1200]}"
        )

    return "\n\n".join(output)
