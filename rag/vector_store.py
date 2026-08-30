from __future__ import annotations

from typing import Any

from config import settings


def _backend() -> str:
    value = str(settings.VECTOR_STORE_BACKEND or "sqlite").strip().lower()
    if value not in {"sqlite", "milvus"}:
        raise ValueError(f"不支持的向量后端：{value}")
    return value


def sync_chunk_embeddings(**kwargs: Any):
    if _backend() == "milvus":
        from rag.milvus_vector_index import sync_chunk_embeddings as implementation
    else:
        from rag.sqlite_vector_index import sync_chunk_embeddings as implementation
    return implementation(**kwargs)


def get_vector_status(**kwargs: Any):
    if _backend() == "milvus":
        from rag.milvus_vector_index import get_vector_status as implementation
    else:
        from rag.sqlite_vector_index import get_vector_status as implementation
    return implementation(**kwargs)


def vector_search(*args: Any, **kwargs: Any):
    if _backend() == "milvus":
        from rag.milvus_vector_index import vector_search as implementation
    else:
        from rag.sqlite_vector_index import vector_search as implementation
    return implementation(*args, **kwargs)


def retrieve_from_vector_store(query: str, top_k: int = 5):
    """兼容旧RAG入口，统一转发到当前向量后端。"""
    return vector_search(query, top_k=top_k)
