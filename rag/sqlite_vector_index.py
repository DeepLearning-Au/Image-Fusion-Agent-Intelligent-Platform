from __future__ import annotations

import hashlib
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import numpy as np

from config import settings
from rag import sqlite_kb


def _db_path(db_path: Optional[Path] = None) -> Path:
    return Path(db_path or sqlite_kb.DB_PATH)


@contextmanager
def _connect(db_path: Optional[Path] = None) -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(str(_db_path(db_path)))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 5000")
    try:
        yield conn
    finally:
        conn.close()


def _content_hash(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8", errors="ignore")).hexdigest()


def _normalize(vector: Any) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float32).reshape(-1)
    length = float(np.linalg.norm(value))
    return value / length if length > 0 else value


def _model_name(embedder: Any, model_name: Optional[str]) -> str:
    return model_name or getattr(embedder, "model", None) or settings.EMBEDDING_MODEL


def _get_embedder(embedder: Any = None) -> Any:
    if embedder is not None:
        return embedder
    from model.factory import get_embedding_model

    return get_embedding_model()


def _has_generation_schema(conn: sqlite3.Connection) -> bool:
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'kb_generations'"
    ).fetchone() is not None


def sync_chunk_embeddings(
    *,
    force: bool = False,
    embedder: Any = None,
    model_name: Optional[str] = None,
    batch_size: Optional[int] = None,
    db_path: Optional[Path] = None,
    generation_id: Optional[int] = None,
) -> Dict[str, Any]:
    """只向量化新增或内容发生变化的 SQLite 文本块。"""
    if db_path is None:
        sqlite_kb.init_db()
    embedder = _get_embedder(embedder)
    model = _model_name(embedder, model_name)
    batch_size = int(batch_size or settings.EMBEDDING_BATCH_SIZE)

    with _connect(db_path) as conn:
        conn.execute("DELETE FROM chunk_embeddings WHERE chunk_id NOT IN (SELECT id FROM chunks)")
        if _has_generation_schema(conn):
            if generation_id is None:
                active = conn.execute(
                    "SELECT id FROM kb_generations WHERE status = 'active' LIMIT 1"
                ).fetchone()
                generation_id = int(active["id"]) if active else -1
            rows = conn.execute(
                """
                SELECT c.id AS chunk_id, c.content, e.content_hash, e.model
                FROM chunks c
                JOIN documents d ON d.id = c.doc_id
                LEFT JOIN chunk_embeddings e ON e.chunk_id = c.id
                WHERE d.generation_id = ?
                ORDER BY c.id
                """,
                (generation_id,),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT c.id AS chunk_id, c.content, e.content_hash, e.model
                FROM chunks c
                LEFT JOIN chunk_embeddings e ON e.chunk_id = c.id
                ORDER BY c.id
                """
            ).fetchall()

    pending = []
    for row in rows:
        content_hash = _content_hash(row["content"])
        if force or row["content_hash"] != content_hash or row["model"] != model:
            pending.append((int(row["chunk_id"]), row["content"], content_hash))

    embedded = []
    for start in range(0, len(pending), batch_size):
        batch = pending[start:start + batch_size]
        vectors = embedder.embed_documents([item[1] for item in batch])
        if len(vectors) != len(batch):
            raise RuntimeError("Embedding 返回数量与文本块数量不一致")
        for (chunk_id, _text, content_hash), vector in zip(batch, vectors):
            normalized = _normalize(vector)
            embedded.append((
                chunk_id,
                content_hash,
                model,
                int(normalized.size),
                normalized.tobytes(),
                sqlite_kb.now_str(),
            ))

    if embedded:
        with _connect(db_path) as conn:
            conn.executemany(
                """
                INSERT INTO chunk_embeddings(
                    chunk_id, content_hash, model, dimensions, vector, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(chunk_id) DO UPDATE SET
                    content_hash = excluded.content_hash,
                    model = excluded.model,
                    dimensions = excluded.dimensions,
                    vector = excluded.vector,
                    updated_at = excluded.updated_at
                """,
                embedded,
            )
            conn.commit()

    status = get_vector_status(
        model_name=model, db_path=db_path, generation_id=generation_id
    )
    return {"embedded_now": len(embedded), **status}


def get_vector_status(
    *,
    model_name: Optional[str] = None,
    db_path: Optional[Path] = None,
    generation_id: Optional[int] = None,
) -> Dict[str, Any]:
    model = model_name or settings.EMBEDDING_MODEL
    with _connect(db_path) as conn:
        if _has_generation_schema(conn):
            if generation_id is None:
                active = conn.execute(
                    "SELECT id FROM kb_generations WHERE status = 'active' LIMIT 1"
                ).fetchone()
                generation_id = int(active["id"]) if active else -1
            total_chunks = int(conn.execute(
                """
                SELECT COUNT(*) FROM chunks c
                JOIN documents d ON d.id = c.doc_id
                WHERE d.generation_id = ?
                """,
                (generation_id,),
            ).fetchone()[0])
            indexed_chunks = int(conn.execute(
                """
                SELECT COUNT(*) FROM chunk_embeddings e
                JOIN chunks c ON c.id = e.chunk_id
                JOIN documents d ON d.id = c.doc_id
                WHERE e.model = ? AND d.generation_id = ?
                """,
                (model, generation_id),
            ).fetchone()[0])
        else:
            total_chunks = int(conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])
            indexed_chunks = int(conn.execute(
                "SELECT COUNT(*) FROM chunk_embeddings WHERE model = ?", (model,)
            ).fetchone()[0])
    return {
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
    auto_sync: bool = True,
    db_path: Optional[Path] = None,
    generation_id: Optional[int] = None,
) -> List[Dict[str, Any]]:
    query = (query or "").strip()
    if not query:
        return []
    if db_path is None:
        sqlite_kb.init_db()
    embedder = _get_embedder(embedder)
    model = _model_name(embedder, model_name)
    if auto_sync:
        sync_chunk_embeddings(
            embedder=embedder,
            model_name=model,
            db_path=db_path,
            generation_id=generation_id,
        )

    query_vector = _normalize(embedder.embed_query(query))
    with _connect(db_path) as conn:
        has_generation = _has_generation_schema(conn)
        generation_join = "JOIN kb_generations g ON g.id = d.generation_id" if has_generation else ""
        if has_generation and generation_id is not None:
            generation_filter = "AND d.generation_id = ?"
            params = (model, int(generation_id))
        elif has_generation:
            generation_filter = "AND g.status = 'active'"
            params = (model,)
        else:
            generation_filter = ""
            params = (model,)
        rows = conn.execute(
            f"""
            SELECT c.id AS chunk_id, c.doc_id, c.parent_id, c.chunk_index, c.content,
                   d.title, d.category, d.source, e.dimensions, e.vector
            FROM chunk_embeddings e
            JOIN chunks c ON c.id = e.chunk_id
            JOIN documents d ON d.id = c.doc_id
            {generation_join}
            WHERE d.enabled = 1 AND e.model = ? {generation_filter}
            """,
            params,
        ).fetchall()

    scored = []
    for row in rows:
        if int(row["dimensions"]) != int(query_vector.size):
            continue
        vector = np.frombuffer(row["vector"], dtype=np.float32)
        similarity = float(np.dot(query_vector, vector))
        scored.append((similarity, row))
    scored.sort(key=lambda item: item[0], reverse=True)

    results = []
    for rank, (score, row) in enumerate(scored[:top_k], start=1):
        results.append({
            "chunk_id": int(row["chunk_id"]),
            "doc_id": int(row["doc_id"]),
            "parent_id": int(row["parent_id"]) if row["parent_id"] is not None else None,
            "chunk_index": int(row["chunk_index"]),
            "doc_title": row["title"],
            "category": row["category"],
            "source": row["source"],
            "text": row["content"],
            "vector_rank": rank,
            "vector_score": round(score, 6),
        })
    return results
