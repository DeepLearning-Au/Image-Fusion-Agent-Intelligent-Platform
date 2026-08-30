import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rag import sqlite_kb


class IncrementalKnowledgeImportTest(unittest.TestCase):
    def test_parent_boundary_uses_level_one_and_two_headings(self):
        hierarchy = sqlite_kb.split_parent_child_chunks(
            "# 产品手册\n\n## 技术参数\n参数说明。\n\n### 分辨率\n640×512。\n\n### 帧率\n50Hz。",
            parent_size=2000,
            child_size=40,
            child_overlap=5,
        )
        self.assertEqual(len(hierarchy), 1)
        self.assertEqual(hierarchy[0]["title_path"], "产品手册 / 技术参数")
        self.assertIn("### 分辨率", hierarchy[0]["content"])

    def test_import_skips_unchanged_and_updates_changed_file(self):
        with tempfile.TemporaryDirectory() as temp_name:
            temp_dir = Path(temp_name)
            db_path = temp_dir / "kb.sqlite3"
            doc_dir = temp_dir / "docs"
            doc_dir.mkdir()
            doc_path = doc_dir / "说明.md"
            doc_path.write_text("第一版设备说明", encoding="utf-8")

            with patch.object(sqlite_kb, "DB_PATH", db_path):
                first = sqlite_kb.import_folder_incremental_report(doc_dir)
                second = sqlite_kb.import_folder_incremental_report(doc_dir)
                doc_path.write_text("第二版设备说明", encoding="utf-8")
                third = sqlite_kb.import_folder_incremental_report(doc_dir)
                status = sqlite_kb.get_status()
                with sqlite_kb.connect() as conn:
                    unlinked_children = conn.execute(
                        "SELECT COUNT(*) FROM chunks WHERE parent_id IS NULL"
                    ).fetchone()[0]

            self.assertEqual(len(first["imported"]), 1)
            self.assertEqual(len(second["skipped"]), 1)
            self.assertEqual(len(third["updated"]), 1)
            self.assertEqual(status["doc_count"], 1)
            self.assertGreaterEqual(status["parent_chunk_count"], 1)
            self.assertEqual(unlinked_children, 0)


if __name__ == "__main__":
    unittest.main()
