import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rag import sqlite_kb
from rag.pdf_parser import (
    OcrPageResult,
    PdfParseError,
    UnavailableOcrProvider,
    analyze_native_text,
    parse_pdf,
)


PROJECT_DIR = Path(__file__).resolve().parents[1]
SAMPLE_DIR = PROJECT_DIR / "data" / "knowledge_docs"


class FakePage:
    def __init__(self, text: str, image_count: int = 0):
        self._text = text
        self.images = [object()] * image_count

    def extract_text(self):
        return self._text


class FakeReader:
    def __init__(self, pages):
        self.pages = pages


class FakeOcrProvider:
    available = True

    def parse_page(self, pdf_path: Path, page_number: int) -> OcrPageResult:
        return OcrPageResult(
            text="红外分辨率 640×512，帧率 50Hz，支持网络视频输出。",
            confidence=0.96,
            engine="fake",
        )


class PdfParserTest(unittest.TestCase):
    def test_native_quality_router(self):
        good = analyze_native_text("红外热成像产品参数说明。" * 12)
        blank = analyze_native_text("")
        garbled = analyze_native_text("产品参数" * 20 + "\ufffd" * 20)

        self.assertFalse(good["needs_ocr"])
        self.assertTrue(blank["needs_ocr"])
        self.assertTrue(garbled["needs_ocr"])

    def test_scanned_page_uses_configured_ocr_provider(self):
        reader = FakeReader([FakePage("", image_count=1)])
        with patch("pypdf.PdfReader", return_value=reader):
            result = parse_pdf(Path("scanned.pdf"), ocr_provider=FakeOcrProvider())

        self.assertEqual(result.status, "completed")
        self.assertEqual(result.ocr_page_count, 1)
        self.assertEqual(result.needs_ocr_page_count, 0)
        self.assertEqual(result.pages[0].parse_method, "ocr:fake")
        self.assertAlmostEqual(result.pages[0].ocr_confidence, 0.96)

    def test_empty_scanned_pdf_fails_closed_without_ocr(self):
        reader = FakeReader([FakePage("", image_count=1)])
        with patch("pypdf.PdfReader", return_value=reader):
            with self.assertRaises(PdfParseError):
                parse_pdf(Path("scanned.pdf"), ocr_provider=UnavailableOcrProvider())

    def test_current_product_pdfs_are_native_text_documents(self):
        # 只验证版本库内的三个核心产品目录；new/用于用户动态上传，不应改变固定样本数量。
        product_dirs = sorted(path for path in SAMPLE_DIR.glob("[0-9][0-9]_*") if path.is_dir())
        pdfs = sorted(pdf for directory in product_dirs for pdf in directory.rglob("*.pdf"))
        self.assertEqual(len(pdfs), 3)
        for pdf in pdfs:
            parsed = parse_pdf(pdf)
            self.assertEqual(parsed.status, "completed", pdf.name)
            self.assertEqual(parsed.native_page_count, parsed.page_count, pdf.name)
            self.assertEqual(parsed.needs_ocr_page_count, 0, pdf.name)

    def test_pdf_report_is_persisted_with_document(self):
        pdf = sorted(SAMPLE_DIR.rglob("*.pdf"))[0]
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "kb.sqlite3"
            with patch.object(sqlite_kb, "DB_PATH", db_path):
                doc_id = sqlite_kb.add_file_document(pdf, category="测试")
                report = sqlite_kb.get_document_parse_report(doc_id)
                status = sqlite_kb.get_status()
                with sqlite_kb.connect() as conn:
                    stored_table_count = conn.execute(
                        "SELECT COUNT(*) FROM document_tables WHERE doc_id = ?",
                        (doc_id,),
                    ).fetchone()[0]

        self.assertIsNotNone(report)
        self.assertEqual(report["parser_name"], "pymupdf_rapidocr_pdfplumber")
        self.assertEqual(report["page_count"], 2)
        self.assertEqual(len(report["pages"]), 2)
        self.assertGreaterEqual(report["table_count"], 1)
        self.assertGreaterEqual(stored_table_count, 1)
        self.assertGreaterEqual(report["extracted_image_count"], 1)
        self.assertEqual(status["needs_ocr_page_count"], 0)


if __name__ == "__main__":
    unittest.main()
