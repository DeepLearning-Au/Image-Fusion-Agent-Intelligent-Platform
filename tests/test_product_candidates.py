import json
import tempfile
import unittest
from pathlib import Path

from catalog.candidates import (
    _candidate_rows,
    _table_modality,
    generate_candidates_from_document_tables,
    get_candidate_status,
    list_candidates,
    publish_approved_candidates,
    review_candidate,
)
from catalog.repository import get_product
from catalog.schema import apply_migrations, connect


PROJECT_DIR = Path(__file__).resolve().parents[1]
SOURCE_PDF = PROJECT_DIR / "data" / "knowledge_docs" / "01_纯红外设备" / "GuideIR_IPT640M纯红外测温机芯.pdf"


class ProductCandidateTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "catalog.sqlite3"
        rows = [
            ["产品型号", "TEST640"],
            ["热成像参数", ""],
            ["红外分辨率", "640×512@12μm"],
            ["SDK/API", "支持软件集成SDK/API"],
        ]
        with connect(self.db_path) as conn:
            apply_migrations(conn)
            conn.execute(
                """
                CREATE TABLE documents (
                    id INTEGER PRIMARY KEY, title TEXT, source TEXT, category TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE document_tables (
                    id INTEGER PRIMARY KEY, doc_id INTEGER, page_number INTEGER,
                    table_index INTEGER, rows_json TEXT
                )
                """
            )
            conn.execute(
                "INSERT INTO documents VALUES (1, ?, ?, '纯红外设备')",
                (SOURCE_PDF.name, str(SOURCE_PDF)),
            )
            conn.execute(
                "INSERT INTO document_tables VALUES (1, 1, 2, 1, ?)",
                (json.dumps(rows, ensure_ascii=False),),
            )
            conn.commit()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_generate_review_and_publish(self):
        result = generate_candidates_from_document_tables(db_path=self.db_path)
        self.assertEqual(result["tables_seen"], 1)
        self.assertIn("TEST640", result["models"])

        candidates = list_candidates(status="pending", model="TEST640", db_path=self.db_path)
        width = next(item for item in candidates if item["spec_key"] == "infrared_resolution_width")
        self.assertEqual(width["value"], 640.0)
        self.assertEqual(width["source_page"], 2)

        review_candidate(width["id"], "approved", db_path=self.db_path)
        published = publish_approved_candidates("TEST640", db_path=self.db_path)
        self.assertEqual(published["published_count"], 1)

        product = get_product("TEST640", db_path=self.db_path)
        specs = {item["spec_key"]: item for item in product["specs"]}
        self.assertEqual(specs["infrared_resolution_width"]["value"], 640.0)
        self.assertEqual(get_candidate_status(self.db_path)["published_count"], 1)

    def test_single_model_technical_indicator_table_is_supported(self):
        rows = [
            ["技术指标", "FC225T", "", ""],
            ["热成像参数", "", "", ""],
            ["传感器分辨率", "256×192", "", ""],
            ["可见光参数", "", "", ""],
            ["传感器分辨率", "2560×1920", "", ""],
        ]
        candidates = list(_candidate_rows({"rows": rows}))
        self.assertTrue(any(item["model"] == "FC225T" for item in candidates))
        self.assertTrue(any(item["spec_key"] == "infrared_resolution" for item in candidates))
        self.assertTrue(any(item["spec_key"] == "visible_resolution" for item in candidates))
        self.assertEqual(_table_modality(rows), "dual_ir_visible")


if __name__ == "__main__":
    unittest.main()
