from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from config import settings
from rag import sqlite_kb
from rag.sqlite_vector_index import (
    _connect as sqlite_vector_connect,
    _get_embedder,
    _model_name,
    _normalize,
    sync_chunk_embeddings as sync_sqlite_embedding_cache,
)


def _milvus_imports():
    try:
        from pymilvus import DataType, MilvusClient
    except ImportError as exc:
        raise RuntimeError(
            "未安装pymilvus，请执行：python -m pip install pymilvus==3.0.1"
        ) from exc
    return MilvusClient, DataType


def _client():
    MilvusClient, _ = _milvus_imports()
    kwargs = {"uri": settings.MILVUS_URI, "timeout": settings.MILVUS_TIMEOUT_SECONDS}
    if settings.MILVUS_TOKEN:
        kwargs["token"] = settings.MILVUS_TOKEN
    return MilvusClient(**kwargs)


def _generation_id(value: Optional[int]) -> int:
    return int(value if value is not None else sqlite_kb.get_active_generation_id())


def _index_type() -> str:
    value = settings.MILVUS_INDEX_TYPE.upper()
    if value not in {"HNSW", "IVF_FLAT"}:
        raise ValueError("MILVUS_INDEX_TYPE仅支持HNSW或IVF_FLAT")
    return value


def _index_params(client):
    index_type = _index_type()
    params = client.prepare_index_params()
    params.add_index(
        field_name="generation_id",
        index_name="generation_id_index",
        index_type="INVERTED",
    )
    params.add_index(
        field_name="embedding_model",
        index_name="embedding_model_index",
        index_type="INVERTED",
    )
    vector_params = (
        {"M": settings.MILVUS_HNSW_M, "efConstruction": settings.MILVUS_HNSW_EF_CONSTRUCTION}
        if index_type == "HNSW"
        else {"nlist": settings.MILVUS_IVF_NLIST}
    )
    params.add_index(
        field_name="vector",
        index_name="vector_index",
        index_type=index_type,
        metric_type=settings.MILVUS_METRIC_TYPE,
        params=vector_params,
    )
    return params


def _vector_index_params(client):
    index_type = _index_type()
    params = client.prepare_index_params()
    vector_params = (
        {"M": settings.MILVUS_HNSW_M, "efConstruction": settings.MILVUS_HNSW_EF_CONSTRUCTION}
        if index_type == "HNSW"
        else {"nlist": settings.MILVUS_IVF_NLIST}
    )
    params.add_index(
        field_name="vector",
        index_name="vector_index",
        index_type=index_type,
        metric_type=settings.MILVUS_METRIC_TYPE,
        params=vector_params,
    )
    return params


def _search_params() -> Dict[str, Any]:
    if _index_type() == "HNSW":
        return {
            "metric_type": settings.MILVUS_METRIC_TYPE,
            "params": {"ef": settings.MILVUS_HNSW_EF_SEARCH},
        }
    return {
        "metric_type": settings.MILVUS_METRIC_TYPE,
        "params": {"nprobe": settings.MILVUS_IVF_NPROBE},
    }


def _truncate(value: Any, max_bytes: int) -> str:
    raw = str(value or "").encode("utf-8")
    if len(raw) <= max_bytes:
        return raw.decode("utf-8")
    return raw[:max_bytes].decode("utf-8", errors="ignore")


def _ensure_collection(client, dimension: int) -> None:
    _, DataType = _milvus_imports()
    name = settings.MILVUS_COLLECTION
    if client.has_collection(collection_name=name):
        description = client.describe_collection(collection_name=name)
        fields = {field["name"]: field for field in description.get("fields", [])}
        vector_field = fields.get("vector", {})
        existing_dim = int(vector_field.get("params", {}).get("dim", dimension))
        if existing_dim != dimension:
            raise RuntimeError(
                f"Milvus集合维度不一致：collection={existing_dim}, embedding={dimension}"
            )
        _ensure_vector_index(client)
        return

    schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field("chunk_id", DataType.INT64, is_primary=True)
    schema.add_field("generation_id", DataType.INT64)
    schema.add_field("doc_id", DataType.INT64)
    schema.add_field("parent_id", DataType.INT64)
    schema.add_field("chunk_index", DataType.INT64)
    schema.add_field("doc_title", DataType.VARCHAR, max_length=1024)
    schema.add_field("category", DataType.VARCHAR, max_length=512)
    schema.add_field("source", DataType.VARCHAR, max_length=4096)
    schema.add_field("embedding_model", DataType.VARCHAR, max_length=256)
    schema.add_field("content_hash", DataType.VARCHAR, max_length=64)
    schema.add_field("content", DataType.VARCHAR, max_length=65535)
    schema.add_field("vector", DataType.FLOAT_VECTOR, dim=dimension)
    client.create_collection(
        collection_name=name,
        schema=schema,
        index_params=_index_params(client),
        timeout=settings.MILVUS_TIMEOUT_SECONDS,
    )


def _ensure_vector_index(client) -> None:
    indexes = client.list_indexes(collection_name=settings.MILVUS_COLLECTION)
    if "vector_index" not in indexes:
        client.create_index(
            collection_name=settings.MILVUS_COLLECTION,
            index_params=_vector_index_params(client),
            timeout=settings.MILVUS_TIMEOUT_SECONDS,
        )
        return
    description = client.describe_index(
        collection_name=settings.MILVUS_COLLECTION,
        index_name="vector_index",
    )
    current = str(description.get("index_type", "")).upper()
    if current and current != _index_type():
        rebuild_vector_index(client=client)


def rebuild_vector_index(*, client: Any = None) -> Dict[str, Any]:
    """按当前配置在线重建HNSW或IVF_FLAT索引，数据本身不删除。"""
    client = client or _client()
    name = settings.MILVUS_COLLECTION
    if not client.has_collection(collection_name=name):
        raise RuntimeError("Milvus集合尚未创建，请先同步向量")
    indexes = client.list_indexes(collection_name=name)
    if "vector_index" in indexes:
        client.release_collection(collection_name=name)
        client.drop_index(
            collection_name=name,
            index_name="vector_index",
            timeout=settings.MILVUS_TIMEOUT_SECONDS,
        )
    client.create_index(
        collection_name=name,
        index_params=_vector_index_params(client),
        timeout=settings.MILVUS_TIMEOUT_SECONDS,
    )
    client.load_collection(
        collection_name=name,
        timeout=settings.MILVUS_TIMEOUT_SECONDS,
    )
    return {
        "backend": "milvus",
        "collection": name,
        "index_type": _index_type(),
        "metric_type": settings.MILVUS_METRIC_TYPE,
        "rebuilt": True,
    }


def _sqlite_rows(
    generation_id: int,
    model: str,
    db_path: Optional[Path],
) -> List[sqlite3.Row]:
    with sqlite_vector_connect(db_path) as conn:
        return conn.execute(
            """
            SELECT c.id AS chunk_id, c.doc_id, c.parent_id, c.chunk_index,
                   c.content, d.title, d.category, d.source,
                   e.content_hash, e.model, e.dimensions, e.vector
            FROM chunks c
            JOIN documents d ON d.id = c.doc_id
            JOIN chunk_embeddings e ON e.chunk_id = c.id
            WHERE d.generation_id = ? AND d.enabled = 1 AND e.model = ?
            ORDER BY c.id
            """,
            (generation_id, model),
        ).fetchall()


def _existing_entities(client, generation_id: int) -> Dict[int, Dict[str, Any]]:
    if not client.has_collection(collection_name=settings.MILVUS_COLLECTION):
        return {}
    rows = client.query(
        collection_name=settings.MILVUS_COLLECTION,
        filter=f"generation_id == {generation_id}",
        output_fields=["chunk_id", "content_hash", "embedding_model"],
        limit=16384,
        timeout=settings.MILVUS_TIMEOUT_SECONDS,
    )
    return {int(row["chunk_id"]): row for row in rows}


def sync_chunk_embeddings(
    *,
    force: bool = False,
    embedder: Any = None,
    model_name: Optional[str] = None,
    batch_size: Optional[int] = None,
    db_path: Optional[Path] = None,
    generation_id: Optional[int] = None,
) -> Dict[str, Any]:
    """先写SQLite嵌入缓存，再以generation为边界幂等同步到Milvus。"""
    generation_id = _generation_id(generation_id)
    embedder = _get_embedder(embedder)
    model = _model_name(embedder, model_name)
    cache_result = sync_sqlite_embedding_cache(
        force=force,
        embedder=embedder,
        model_name=model,
        batch_size=batch_size,
        db_path=db_path,
        generation_id=generation_id,
    )
    rows = _sqlite_rows(generation_id, model, db_path)
    if not rows:
        return {
            "backend": "milvus", "embedding_model": model,
            "generation_id": generation_id, "embedded_now": cache_result.get("embedded_now", 0),
            "milvus_upserted_now": 0, "total_chunks": 0, "indexed_chunks": 0,
            "pending_chunks": 0, "ready": False,
        }

    dimension = int(rows[0]["dimensions"])
    client = _client()
    _ensure_collection(client, dimension)
    existing = _existing_entities(client, generation_id)
    current_ids = {int(row["chunk_id"]) for row in rows}
    stale_ids = sorted(set(existing) - current_ids)
    for start in range(0, len(stale_ids), 500):
        batch_ids = stale_ids[start:start + 500]
        client.delete(
            collection_name=settings.MILVUS_COLLECTION,
            filter=f"chunk_id in {batch_ids}",
            timeout=settings.MILVUS_TIMEOUT_SECONDS,
        )

    pending = []
    for row in rows:
        chunk_id = int(row["chunk_id"])
        old = existing.get(chunk_id, {})
        if force or old.get("content_hash") != row["content_hash"] or old.get("embedding_model") != model:
            vector = np.frombuffer(row["vector"], dtype=np.float32)
            if vector.size != dimension:
                raise RuntimeError(f"chunk {chunk_id}向量维度异常")
            pending.append({
                "chunk_id": chunk_id,
                "generation_id": generation_id,
                "doc_id": int(row["doc_id"]),
                "parent_id": int(row["parent_id"]) if row["parent_id"] is not None else -1,
                "chunk_index": int(row["chunk_index"]),
                "doc_title": _truncate(row["title"], 1024),
                "category": _truncate(row["category"], 512),
                "source": _truncate(row["source"], 4096),
                "embedding_model": _truncate(model, 256),
                "content_hash": row["content_hash"],
                "content": _truncate(row["content"], 65535),
                "vector": vector.tolist(),
            })

    size = int(settings.MILVUS_UPSERT_BATCH_SIZE)
    for start in range(0, len(pending), size):
        client.upsert(
            collection_name=settings.MILVUS_COLLECTION,
            data=pending[start:start + size],
            timeout=settings.MILVUS_TIMEOUT_SECONDS,
        )
    if pending or stale_ids:
        client.flush(collection_name=settings.MILVUS_COLLECTION)
    client.load_collection(
        collection_name=settings.MILVUS_COLLECTION,
        timeout=settings.MILVUS_TIMEOUT_SECONDS,
    )
    status = get_vector_status(model_name=model, generation_id=generation_id)
    return {
        "embedded_now": cache_result.get("embedded_now", 0),
        "milvus_upserted_now": len(pending),
        "milvus_deleted_now": len(stale_ids),
        **status,
    }


def get_vector_status(
    *,
    model_name: Optional[str] = None,
    db_path: Optional[Path] = None,
    generation_id: Optional[int] = None,
) -> Dict[str, Any]:
    model = model_name or settings.EMBEDDING_MODEL
    generation_id = _generation_id(generation_id)
    with sqlite_vector_connect(db_path) as conn:
        total_chunks = int(conn.execute(
            """
            SELECT COUNT(*) FROM chunks c JOIN documents d ON d.id = c.doc_id
            WHERE d.generation_id = ? AND d.enabled = 1
            """,
            (generation_id,),
        ).fetchone()[0])
    client = _client()
    indexed_chunks = 0
    if client.has_collection(collection_name=settings.MILVUS_COLLECTION):
        result = client.query(
            collection_name=settings.MILVUS_COLLECTION,
            filter=f'generation_id == {generation_id} and embedding_model == "{model}"',
            output_fields=["count(*)"],
            timeout=settings.MILVUS_TIMEOUT_SECONDS,
        )
        if result:
            indexed_chunks = int(result[0].get("count(*)", 0))
    return {
        "backend": "milvus",
        "uri": settings.MILVUS_URI,
        "collection": settings.MILVUS_COLLECTION,
        "index_type": _index_type(),
        "metric_type": settings.MILVUS_METRIC_TYPE,
        "embedding_model": model,
        "generation_id": generation_id,
        "total_chunks": total_chunks,
        "indexed_chunks": indexed_chunks,
        "pending_chunks": max(total_chunks - indexed_chunks, 0),
        "ready": total_chunks > 0 and indexed_chunks == total_chunks,
    }


def vector_search(
    query: str,
    *,
    top_k: int = 20,
    embedder: Any = None,
    model_name: Optional[str] = None,
    auto_sync: bool = False,
    db_path: Optional[Path] = None,
    generation_id: Optional[int] = None,
) -> List[Dict[str, Any]]:
    query = (query or "").strip()
    if not query:
        return []
    generation_id = _generation_id(generation_id)
    embedder = _get_embedder(embedder)
    model = _model_name(embedder, model_name)
    if auto_sync:
        sync_chunk_embeddings(
            embedder=embedder,
            model_name=model,
            db_path=db_path,
            generation_id=generation_id,
        )
    query_vector = _normalize(embedder.embed_query(query)).tolist()
    client = _client()
    results = client.search(
        collection_name=settings.MILVUS_COLLECTION,
        data=[query_vector],
        anns_field="vector",
        filter=f'generation_id == {generation_id} and embedding_model == "{model}"',
        limit=top_k,
        output_fields=[
            "chunk_id", "doc_id", "parent_id", "chunk_index", "doc_title",
            "category", "source", "content",
        ],
        search_params=_search_params(),
        timeout=settings.MILVUS_TIMEOUT_SECONDS,
    )
    hits = results[0] if results else []
    output = []
    for rank, hit in enumerate(hits, start=1):
        entity = hit.get("entity", {})
        parent_id = int(entity.get("parent_id", -1))
        output.append({
            "chunk_id": int(entity.get("chunk_id", hit.get("id"))),
            "doc_id": int(entity["doc_id"]),
            "parent_id": parent_id if parent_id >= 0 else None,
            "chunk_index": int(entity["chunk_index"]),
            "doc_title": entity.get("doc_title", ""),
            "category": entity.get("category", ""),
            "source": entity.get("source", ""),
            "text": entity.get("content", ""),
            "vector_rank": rank,
            "vector_score": round(float(hit.get("distance", 0.0)), 6),
            "vector_backend": "milvus",
            "vector_index_type": _index_type(),
        })
    return output
