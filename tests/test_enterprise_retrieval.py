import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rag.enterprise_retrieval import expand_parent_context, reciprocal_rank_fusion
from rag.bge_reranker import rerank_candidates
from rag.sqlite_vector_index import sync_chunk_embeddings, vector_search


class FakeEmbeddings:
    model = "fake-embedding-v1"

    @staticmethod
    def _vector(text):
        if any(word in text for word in ["夜间", "低照度", "热目标"]):
            return [1.0, 0.0, 0.0]
        if any(word in text for word in ["白天", "纹理", "颜色"]):
            return [0.0, 1.0, 0.0]
        return [0.0, 0.0, 1.0]

    def embed_documents(self, texts):
        return [self._vector(text) for text in texts]

    def embed_query(self, text):
        return self._vector(text)


class FakeReranker:
    model_name = "fake-bge-reranker"

    def score_pairs(self, query, texts):
        return [0.95 if "夜间" in text else 0.15 for text in texts]


class EnterpriseRetrievalTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "kb.sqlite3"
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript("""
                PRAGMA foreign_keys = ON;
                CREATE TABLE documents(
                    id INTEGER PRIMARY KEY, title TEXT, category TEXT,
                    source TEXT, enabled INTEGER
                );
                CREATE TABLE chunks(
                    id INTEGER PRIMARY KEY, doc_id INTEGER, parent_id INTEGER,
                    chunk_index INTEGER,
                    content TEXT, FOREIGN KEY(doc_id) REFERENCES documents(id)
                );
                CREATE TABLE chunk_embeddings(
                    chunk_id INTEGER PRIMARY KEY, content_hash TEXT NOT NULL,
                    model TEXT NOT NULL, dimensions INTEGER NOT NULL,
                    vector BLOB NOT NULL, updated_at TEXT,
                    FOREIGN KEY(chunk_id) REFERENCES chunks(id) ON DELETE CASCADE
                );
                INSERT INTO documents VALUES(1, '红外设备手册', '纯红外设备', 'ir.pdf', 1);
                INSERT INTO documents VALUES(2, '可见光设备手册', '纯可见光设备', 'rgb.pdf', 1);
                INSERT INTO chunks VALUES(11, 1, NULL, 0, '适合夜间发现热目标');
                INSERT INTO chunks VALUES(22, 2, NULL, 0, '适合白天保留颜色和纹理');
            """)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_sqlite_embeddings_are_incremental_and_searchable(self):
        embedder = FakeEmbeddings()
        first = sync_chunk_embeddings(
            embedder=embedder, model_name=embedder.model, db_path=self.db_path
        )
        second = sync_chunk_embeddings(
            embedder=embedder, model_name=embedder.model, db_path=self.db_path
        )
        self.assertEqual(first["embedded_now"], 2)
        self.assertEqual(second["embedded_now"], 0)

        hits = vector_search(
            "低照度环境发现人员",
            embedder=embedder,
            model_name=embedder.model,
            db_path=self.db_path,
        )
        self.assertEqual(hits[0]["chunk_id"], 11)

    def test_rrf_combines_sparse_and_vector_ranks(self):
        sparse = [{"chunk_id": 11, "text": "A"}, {"chunk_id": 22, "text": "B"}]
        dense = [{"chunk_id": 22, "text": "B"}, {"chunk_id": 11, "text": "A"}]
        fused = reciprocal_rank_fusion([sparse, dense])
        self.assertEqual({item["chunk_id"] for item in fused}, {11, 22})
        self.assertTrue(all(len(item["retrieval_routes"]) == 2 for item in fused))

    def test_bge_reranker_reorders_rrf_candidates(self):
        candidates = [
            {"chunk_id": 22, "text": "白天颜色纹理", "rrf_score": 0.9},
            {"chunk_id": 11, "text": "夜间热目标", "rrf_score": 0.7},
        ]
        ranked = rerank_candidates(
            "夜间应该选择什么设备",
            candidates,
            reranker=FakeReranker(),
        )
        self.assertEqual(ranked[0]["chunk_id"], 11)
        self.assertEqual(ranked[0]["reranker_rank"], 1)
        self.assertIn("reranker_score", ranked[0])

    def test_parent_lookup_replaces_child_text_and_deduplicates(self):
        children = [
            {"chunk_id": 1, "parent_id": 10, "text": "参数子块", "final_score": 0.9},
            {"chunk_id": 2, "parent_id": 10, "text": "场景子块", "final_score": 0.8},
        ]
        parents = [{
            "parent_id": 10,
            "parent_index": 0,
            "title_path": "产品说明 / 技术参数",
            "content": "完整的产品说明章节",
            "doc_title": "设备手册.md",
            "category": "设备资料",
            "source": "manual",
        }]
        with patch("rag.enterprise_retrieval.sqlite_kb.get_parent_chunks", return_value=parents):
            expanded = expand_parent_context(children)
        self.assertEqual(len(expanded), 1)
        self.assertEqual(expanded[0]["text"], "完整的产品说明章节")
        self.assertEqual(expanded[0]["matched_child_ids"], [1, 2])


if __name__ == "__main__":
    unittest.main()
