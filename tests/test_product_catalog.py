import tempfile
import unittest
from pathlib import Path

from catalog.repository import (
    apply_product_profile,
    get_catalog_status,
    get_product,
    import_product_seed,
    search_products,
)


PROJECT_DIR = Path(__file__).resolve().parents[1]
SEED_PATH = PROJECT_DIR / "data" / "product_seeds" / "ipt640m.json"


class ProductCatalogTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "catalog.sqlite3"

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_import_is_idempotent_and_keeps_typed_values(self):
        import_product_seed(SEED_PATH, db_path=self.db_path)
        import_product_seed(SEED_PATH, db_path=self.db_path)

        status = get_catalog_status(db_path=self.db_path)
        self.assertEqual(status["product_count"], 1)
        self.assertEqual(status["manufacturer_count"], 1)
        self.assertEqual(status["price_record_count"], 1)
        self.assertEqual(status["media_count"], 1)

        product = get_product("IPT640M", db_path=self.db_path)
        self.assertEqual(product["modality"], "pure_infrared")
        specs = {item["spec_key"]: item for item in product["specs"]}
        self.assertEqual(specs["resolution_width"]["value"], 640.0)
        self.assertEqual(specs["resolution_height"]["value"], 512.0)
        self.assertEqual(specs["stream_frame_rate"]["value"], 25.0)
        self.assertIs(specs["sdk_api_support"]["value"], True)
        self.assertEqual(specs["infrared_resolution"]["source_page"], 2)

        compatibility = product["fusion_compatibility"]
        self.assertEqual(compatibility["mambadfuse_role"], "infrared_source")
        self.assertIsNone(compatibility["hardware_trigger"])
        self.assertEqual(compatibility["suitability_status"], "conditional")

    def test_structured_filter_uses_numeric_and_boolean_fields(self):
        import_product_seed(SEED_PATH, db_path=self.db_path)

        matches = search_products(
            modality="pure_infrared",
            min_width=640,
            min_height=512,
            min_fps=25,
            sdk_required=True,
            db_path=self.db_path,
        )
        self.assertEqual([item["model"] for item in matches], ["IPT640M"])

        no_matches = search_products(
            modality="pure_infrared",
            min_width=1024,
            db_path=self.db_path,
        )
        self.assertEqual(no_matches, [])

    def test_structured_filter_uses_public_price_range(self):
        import_product_seed(SEED_PATH, db_path=self.db_path)
        apply_product_profile(
            {
                "model": "IPT640M",
                "name": "IPT640M迷你网络型测温机芯",
                "modality": "pure_infrared",
                "summary": "测试公开价格筛选",
                "price": {
                    "price_type": "public_list",
                    "amount_min": 8000,
                    "amount_max": 10000,
                    "currency": "CNY",
                    "source_page": 2,
                },
            },
            db_path=self.db_path,
        )

        within_budget = search_products(
            max_price=9000, currency="CNY", db_path=self.db_path
        )
        self.assertEqual([item["model"] for item in within_budget], ["IPT640M"])
        below_range = search_products(
            max_price=7000, currency="CNY", db_path=self.db_path
        )
        self.assertEqual(below_range, [])

    def test_apply_product_profile_keeps_auditable_relations(self):
        import_product_seed(SEED_PATH, db_path=self.db_path)
        apply_product_profile(
            {
                "model": "IPT640M",
                "name": "IPT640M审核产品",
                "modality": "pure_infrared",
                "summary": "审核后的简介",
                "price": {"price_type": "inquiry", "source_page": 2, "note": "需询价"},
                "media": [{"uri": "/knowledge-assets/products/ipt640m_page1.png", "source_page": 1}],
                "scenarios": [{"scenario": "电力巡检", "source_page": 1}],
                "fusion_compatibility": {
                    "mambadfuse_role": "infrared_source",
                    "infrared_stream_access": True,
                    "visible_stream_access": False,
                    "independent_modal_streams": False,
                    "registration_required": True,
                    "suitability_status": "conditional",
                    "source_page": 2,
                },
            },
            db_path=self.db_path,
        )

        product = get_product("IPT640M", db_path=self.db_path)
        self.assertEqual(product["name"], "IPT640M审核产品")
        self.assertEqual(len(product["prices"]), 1)
        self.assertEqual(len(product["media"]), 1)
        self.assertEqual(len(product["scenarios"]), 1)
        self.assertEqual(product["fusion_compatibility"]["mambadfuse_role"], "infrared_source")


if __name__ == "__main__":
    unittest.main()
