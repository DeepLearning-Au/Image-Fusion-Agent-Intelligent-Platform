import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rag import sqlite_kb


class BM25SearchTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "kb.sqlite3"

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_fts5_uses_native_bm25_ranking(self):
        with patch.object(sqlite_kb, "DB_PATH", self.db_path):
            sqlite_kb.init_db()
            with sqlite_kb.connect() as conn:
                if not sqlite_kb.has_fts_table(conn):
                    self.skipTest("当前SQLite没有启用FTS5")

            sqlite_kb.add_text_document(
                "红外相机SDK产品",
                "红外相机SDK支持二次开发，并提供完整接口。红外相机SDK。",
                category="红外设备",
            )
            sqlite_kb.add_text_document(
                "通用成像设备",
                "本设备包含基础红外相机功能。",
                category="成像设备",
            )

            result = sqlite_kb.search_knowledge("红外相机SDK", top_k=5)

            self.assertEqual(result["retrieval"]["ranker"], "bm25")
            self.assertEqual(result["sources"][0]["doc_title"], "红外相机SDK产品")
            self.assertEqual(result["sources"][0]["sparse_retriever"], "fts5_bm25")
            self.assertIsNotNone(result["sources"][0]["bm25_raw_score"])

    def test_two_character_chinese_query_keeps_like_fallback(self):
        with patch.object(sqlite_kb, "DB_PATH", self.db_path):
            sqlite_kb.init_db()
            sqlite_kb.add_text_document("红外设备", "红外探测器产品介绍")

            result = sqlite_kb.search_knowledge("红外", top_k=5)

            self.assertTrue(result["sources"])
            self.assertEqual(result["retrieval"]["method"], "like_fallback")


if __name__ == "__main__":
    unittest.main()
