import unittest
from unittest.mock import patch

from rag import milvus_vector_index


class MilvusIndexConfigurationTest(unittest.TestCase):
    def test_hnsw_search_parameters(self):
        with patch.object(milvus_vector_index.settings, "MILVUS_INDEX_TYPE", "HNSW"):
            params = milvus_vector_index._search_params()
        self.assertEqual(params["params"]["ef"], milvus_vector_index.settings.MILVUS_HNSW_EF_SEARCH)

    def test_ivf_search_parameters(self):
        with patch.object(milvus_vector_index.settings, "MILVUS_INDEX_TYPE", "IVF_FLAT"):
            params = milvus_vector_index._search_params()
        self.assertEqual(params["params"]["nprobe"], milvus_vector_index.settings.MILVUS_IVF_NPROBE)

    def test_unknown_index_is_rejected(self):
        with patch.object(milvus_vector_index.settings, "MILVUS_INDEX_TYPE", "UNKNOWN"):
            with self.assertRaisesRegex(ValueError, "HNSW或IVF_FLAT"):
                milvus_vector_index._index_type()


if __name__ == "__main__":
    unittest.main()
