import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rag import sqlite_kb
from rag.generation_service import (
    build_generation,
    list_generations,
    publish_generation,
    rollback_generation,
)


class FakeEmbeddings:
    model = "fake-generation-embedding"

    def embed_documents(self, texts):
        return [[1.0, float(index + 1), 0.5] for index, _ in enumerate(texts)]

    def embed_query(self, text):
        return [1.0, 1.0, 0.5]


class FailingEmbeddings(FakeEmbeddings):
    def embed_documents(self, texts):
        raise RuntimeError("模拟Embedding失败")


class KnowledgeGenerationTest(unittest.TestCase):
    def setUp(self):
        self.backend_patch = patch(
            "rag.vector_store.settings.VECTOR_STORE_BACKEND", "sqlite"
        )
        self.backend_patch.start()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "kb.sqlite3"
        self.docs_dir = Path(self.temp_dir.name) / "docs"
        self.docs_dir.mkdir()
        (self.docs_dir / "新版说明.md").write_text(
            "# 新版知识\n\n## 产品能力\n新版独有的夜间红外检测能力。",
            encoding="utf-8",
        )

    def tearDown(self):
        self.temp_dir.cleanup()
        self.backend_patch.stop()

    def test_atomic_publish_and_manual_rollback(self):
        with patch.object(sqlite_kb, "DB_PATH", self.db_path):
            sqlite_kb.init_db()
            old_generation_id = sqlite_kb.get_active_generation_id()
            sqlite_kb.add_text_document("旧版说明", "旧版独有内容")
            draft = build_generation(
                folders=[self.docs_dir], embedder=FakeEmbeddings(), publish=False,
                evaluation_enabled=False,
            )

            def fail_during_switch():
                raise RuntimeError("模拟发布中断")

            with self.assertRaisesRegex(RuntimeError, "模拟发布中断"):
                publish_generation(
                    draft["generation_id"],
                    require_gate=False,
                    _before_activate=fail_during_switch,
                )
            self.assertEqual(sqlite_kb.get_active_generation_id(), old_generation_id)

            publish_generation(draft["generation_id"], require_gate=False)
            self.assertEqual(
                sqlite_kb.get_active_generation_id(), draft["generation_id"]
            )
            self.assertTrue(sqlite_kb.search_knowledge("新版独有")["sources"])

            rollback_generation(old_generation_id)
            self.assertEqual(sqlite_kb.get_active_generation_id(), old_generation_id)
            self.assertTrue(sqlite_kb.search_knowledge("旧版独有")["sources"])

    def test_build_failure_keeps_active_generation(self):
        with patch.object(sqlite_kb, "DB_PATH", self.db_path):
            sqlite_kb.init_db()
            active_before = sqlite_kb.get_active_generation_id()
            with self.assertRaisesRegex(RuntimeError, "模拟Embedding失败"):
                build_generation(
                    folders=[self.docs_dir],
                    embedder=FailingEmbeddings(),
                    publish=True,
                    evaluation_enabled=False,
                )
            self.assertEqual(sqlite_kb.get_active_generation_id(), active_before)
            self.assertEqual(list_generations()[0]["status"], "failed")


if __name__ == "__main__":
    unittest.main()
