import tempfile
import unittest
from pathlib import Path

from catalog.repository import import_product_seed
from rag.device_query import parse_device_query
from rag.hybrid_product_search import hybrid_product_search


PROJECT_DIR = Path(__file__).resolve().parents[1]
SEED_PATH = PROJECT_DIR / "data" / "product_seeds" / "ipt640m.json"


class FakeProductReranker:
    model_name = "fake-product-bge"

    def score_pairs(self, query, texts):
        return [0.91 for _ in texts]


class HybridProductSearchTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "hybrid.sqlite3"
        import_product_seed(SEED_PATH, db_path=self.db_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_query_parser_extracts_hard_conditions(self):
        parsed = parse_device_query(
            "推荐纯红外640×512、25Hz、支持ONVIF和SDK并适合MambaDFuse融合的设备",
            known_models=["IPT640M"],
        )
        self.assertEqual(parsed.query_type, "recommendation")
        self.assertEqual(parsed.modality, "pure_infrared")
        self.assertEqual((parsed.min_width, parsed.min_height), (640, 512))
        self.assertEqual(parsed.min_fps, 25.0)
        self.assertIn("ONVIF", parsed.required_protocols)
        self.assertIs(parsed.sdk_required, True)
        self.assertTrue(parsed.fusion_required)

        commercial = parse_device_query(
            "预算不超过1.5万元，必须支持RTSP，USB3 Vision优先"
        )
        self.assertEqual(commercial.max_price, 15000)
        self.assertEqual(commercial.currency, "CNY")
        self.assertTrue(commercial.price_constraint_hard)
        self.assertIn("RTSP", commercial.required_protocols)
        self.assertIn("USB3 Vision", commercial.preferred_protocols)

        optional = parse_device_query("推荐设备，无需SDK，也不需要融合")
        self.assertIsNone(optional.sdk_required)
        self.assertFalse(optional.fusion_required)

    def test_query_parser_corrects_protocol_typos_but_not_risky_numbers(self):
        typo = parse_device_query("必须支持USB3 Vison的工业相机")
        self.assertIn("USB3 Vision", typo.required_protocols)
        self.assertNotIn("USB 3.0", typo.required_protocols)
        self.assertEqual(typo.corrections[0]["corrected"], "USB3 Vision")
        self.assertEqual(typo.corrections[0]["applied_as"], "hard_filter")
        self.assertGreaterEqual(
            typo.constraint_confidence["protocol:USB3 Vision"], 0.90
        )

        suspicious = parse_device_query("推荐分辨率至少640×51的红外设备")
        self.assertIsNone(suspicious.min_width)
        self.assertIsNone(suspicious.min_height)
        self.assertTrue(suspicious.needs_confirmation)
        self.assertEqual(suspicious.ambiguous_constraints[0]["kind"], "resolution")

    def test_model_typo_is_suggested_but_not_silently_filtered(self):
        parsed = parse_device_query(
            "推荐IPT640N用于电力巡检",
            known_models=["IPT640M"],
        )
        self.assertIsNone(parsed.model)
        self.assertTrue(parsed.needs_confirmation)
        self.assertEqual(parsed.ambiguous_constraints[0]["kind"], "model")
        self.assertEqual(parsed.ambiguous_constraints[0]["suggested"], "IPT640M")

    def test_hybrid_search_uses_sql_for_hard_filters(self):
        result = hybrid_product_search(
            "推荐纯红外640×512、25Hz、支持ONVIF和SDK并适合融合的设备",
            db_path=self.db_path,
        )
        self.assertEqual(result["structured_match_count"], 1)
        self.assertEqual(result["products"][0]["model"], "IPT640M")
        self.assertIn("硬件同步", result["products"][0]["needs_vendor_confirmation"])

        no_match = hybrid_product_search(
            "推荐纯红外1024×768设备",
            db_path=self.db_path,
        )
        self.assertEqual(no_match["structured_match_count"], 0)
        self.assertIn("没有产品同时满足全部硬条件", no_match["warnings"])
        resolution_relaxation = next(
            item for item in no_match["relaxation_suggestions"]
            if item["constraint"] == "resolution"
        )
        self.assertEqual(resolution_relaxation["match_count"], 1)
        self.assertEqual(resolution_relaxation["models"], ["IPT640M"])

    def test_product_evidence_is_linked_then_reranked(self):
        result = hybrid_product_search(
            "推荐适合电力巡检的IPT640M",
            db_path=self.db_path,
            rag_result={
                "sources": [{
                    "doc_title": "GuideIR_IPT640M纯红外测温机芯.pdf",
                    "text": "IPT640M适用于电力巡检机器人",
                    "final_score": 0.88,
                }],
                "confidence": "high",
                "top_score": 0.88,
                "retrieval": {"reranker_used": True},
            },
            product_reranker=FakeProductReranker(),
            enable_product_reranker=True,
        )
        product = result["products"][0]
        self.assertEqual(product["ranking"]["bge_score"], 0.91)
        self.assertEqual(product["ranking"]["evidence_count"], 1)
        self.assertTrue(result["retrieval"]["product_reranker_used"])
        self.assertEqual(result["retrieval"]["evidence_linked_product_count"], 1)

if __name__ == "__main__":
    unittest.main()
