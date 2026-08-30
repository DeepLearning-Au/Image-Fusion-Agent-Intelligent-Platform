import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rag import sqlite_kb
from rag.eval_service import evaluate_retrieval
from rag.generation_service import (
    check_evaluation_gate,
    create_generation,
    evaluate_generation,
    publish_generation,
)


class RagEvaluationGateTest(unittest.TestCase):
    def test_metrics_include_ndcg(self):
        items = [{
            "id": "q1", "question": "测试", "intent": "测试",
            "expected_docs": ["目标.md"], "keywords": ["答案"],
        }]
        report = evaluate_retrieval(
            items,
            lambda _: {"sources": [
                {"doc_title": "其他.md", "text": "无"},
                {"doc_title": "目标.md", "text": "答案"},
            ]},
            top_k=5,
        )
        self.assertEqual(report["recall_at_k"], 1.0)
        self.assertEqual(report["mrr"], 0.5)
        self.assertGreater(report["ndcg_at_k"], 0.0)
        self.assertLess(report["ndcg_at_k"], 1.0)

    def test_gate_rejects_low_metrics(self):
        gate = check_evaluation_gate(
            {"total": 30, "recall_at_k": 0.7, "mrr": 0.8, "ndcg_at_k": 0.8},
            {"recall_at_k": 0.8, "mrr": 0.65, "ndcg_at_k": 0.7},
            30,
        )
        self.assertFalse(gate["passed"])
        self.assertIn("recall_at_k", gate["failures"][0])

    def test_persisted_pass_allows_publish(self):
        with tempfile.TemporaryDirectory() as folder:
            db_path = Path(folder) / "kb.sqlite3"
            with patch.object(sqlite_kb, "DB_PATH", db_path):
                sqlite_kb.init_db()
                generation = create_generation("evaluation-test")
                generation_id = generation["id"]
                with sqlite_kb.connect() as conn:
                    conn.execute(
                        "UPDATE kb_generations SET status='ready' WHERE id=?",
                        (generation_id,),
                    )
                    conn.commit()
                items = [{
                    "id": "q1", "question": "测试", "intent": "测试",
                    "expected_docs": ["目标.md"], "keywords": ["答案"],
                }]
                report = evaluate_generation(
                    generation_id,
                    eval_items=items,
                    thresholds={"recall_at_k": 1.0, "mrr": 1.0, "ndcg_at_k": 1.0},
                    min_questions=1,
                    search_fn=lambda _: {"sources": [
                        {"doc_title": "目标.md", "text": "答案"}
                    ]},
                )
                self.assertTrue(report["gate"]["passed"])
                with patch(
                    "rag.generation_service.settings.RAG_ANSWER_EVALUATION_ENABLED",
                    False,
                ):
                    publish_generation(generation_id)
                self.assertEqual(sqlite_kb.get_active_generation_id(), generation_id)


if __name__ == "__main__":
    unittest.main()
