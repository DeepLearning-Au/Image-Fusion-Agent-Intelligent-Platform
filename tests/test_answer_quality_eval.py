import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rag import sqlite_kb
from rag.answer_eval_service import (
    check_answer_quality_gate,
    evaluate_answer_quality,
)
from rag.generation_service import (
    create_generation,
    evaluate_generation_answers,
    get_generation_answer_evaluation,
)


class AnswerQualityEvalTest(unittest.TestCase):
    def setUp(self):
        self.items = [
            {
                "id": "answerable", "question": "参数是什么？", "answerable": True,
                "expected_points": ["640×512"], "expected_docs": ["产品.pdf"],
            },
            {
                "id": "refusal", "question": "价格是多少？", "answerable": False,
                "expected_points": ["资料未提供"], "expected_docs": [],
            },
        ]

    @staticmethod
    def search(_):
        return {"sources": [{"doc_title": "产品.pdf", "text": "分辨率640×512"}]}

    @staticmethod
    def answer(question, _sources):
        return "分辨率为640×512 [1]" if "参数" in question else "当前资料未提供价格，无法确认。"

    @staticmethod
    def judge(item, _answer, _sources):
        return {
            "answer_relevance": 1.0,
            "faithfulness": 1.0,
            "citation_completeness": 1.0,
            "refusal_correctness": 1.0,
            "reason": "符合要求",
        }

    def test_four_quality_metrics(self):
        report = evaluate_answer_quality(
            self.items, self.search, answer_fn=self.answer, judge_fn=self.judge
        )
        self.assertEqual(report["answer_relevance"], 1.0)
        self.assertEqual(report["faithfulness"], 1.0)
        self.assertEqual(report["citation_completeness"], 1.0)
        self.assertEqual(report["refusal_correctness"], 1.0)

    def test_invalid_citation_is_rejected(self):
        report = evaluate_answer_quality(
            self.items[:1], self.search,
            answer_fn=lambda _q, _s: "分辨率为640×512 [9]",
            judge_fn=self.judge,
        )
        self.assertEqual(report["citation_completeness"], 0.0)

    def test_gate_and_report_persistence(self):
        with tempfile.TemporaryDirectory() as folder:
            db_path = Path(folder) / "kb.sqlite3"
            with patch.object(sqlite_kb, "DB_PATH", db_path):
                sqlite_kb.init_db()
                generation_id = create_generation("answer-eval-test")["id"]
                report = evaluate_generation_answers(
                    generation_id,
                    eval_items=self.items,
                    thresholds={
                        "answer_relevance": 1.0,
                        "faithfulness": 1.0,
                        "citation_completeness": 1.0,
                        "refusal_correctness": 1.0,
                    },
                    min_questions=2,
                    search_fn=self.search,
                    answer_fn=self.answer,
                    judge_fn=self.judge,
                )
                self.assertTrue(report["gate"]["passed"])
                stored = get_generation_answer_evaluation(generation_id)
                self.assertTrue(stored["passed"])

    def test_gate_rejects_hallucination(self):
        gate = check_answer_quality_gate(
            {
                "total": 12, "answer_relevance": 0.9, "faithfulness": 0.6,
                "citation_completeness": 0.9, "refusal_correctness": 1.0,
            },
            {
                "answer_relevance": 0.8, "faithfulness": 0.85,
                "citation_completeness": 0.8, "refusal_correctness": 0.9,
            },
            12,
        )
        self.assertFalse(gate["passed"])
        self.assertIn("faithfulness", gate["failures"][0])


if __name__ == "__main__":
    unittest.main()
