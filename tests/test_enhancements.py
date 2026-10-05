"""Unit tests for newly added cross-backbone clients, oracle extractor, and statistical tools."""

import unittest
from unittest.mock import patch, MagicMock

from src.data_loader import Column, ForeignKey, Schema, Table
from src.graph.oracle_join_path import OracleJoinPathExtractor
from src.llm.llm_factory import LLMFactory
from src.llm.mock_client import MockLLMClient
from src.llm.gemini_client import GeminiClient
from src.llm.openai_client import OpenAIClient
from src.evaluator.statistical_test import mcnemar_test, compute_bootstrap_ci


class EnhancementTests(unittest.TestCase):
    def setUp(self):
        self.t_student = Table(0, "student", "student")
        self.c_id = Column(0, "id", "id", 0, "int", is_primary_key=True)
        self.c_name = Column(1, "name", "name", 0, "text")
        self.t_student.columns = [self.c_id, self.c_name]

        self.t_course = Table(1, "course", "course")
        self.c_cid = Column(2, "course_id", "course_id", 1, "int", is_primary_key=True)
        self.t_course.columns = [self.c_cid]

        self.schema = Schema("school", [self.t_student, self.t_course], [], [self.c_id, self.c_name, self.c_cid])

    def test_oracle_join_path_extraction(self):
        sql = "SELECT student.name, course.course_id FROM student JOIN course ON student.id = course.student_id"
        tables = OracleJoinPathExtractor.extract_tables(sql)
        self.assertIn("student", tables)
        self.assertIn("course", tables)

        path = OracleJoinPathExtractor.get_oracle_join_path(sql, self.schema)
        self.assertIsNotNone(path)
        self.assertEqual(2, len(path.path_table_ids))

    def test_llm_factory_instantiation(self):
        mock_client = LLMFactory.create(provider="mock")
        self.assertIsInstance(mock_client, MockLLMClient)

        gemini_client = LLMFactory.create(provider="gemini")
        self.assertIsInstance(gemini_client, GeminiClient)

        openai_client = LLMFactory.create(provider="openai")
        self.assertIsInstance(openai_client, OpenAIClient)

    def test_mock_llm_generation(self):
        client = MockLLMClient()
        prompt = "Question: Find students\nTables: student, course\nColumns: student.id, student.name\nJoin Path: student -> course\n"
        sql = client.generate_sql(prompt)
        self.assertTrue(sql.lower().startswith("select"))
        self.assertIn("from student", sql.lower())

    def test_mcnemar_paired_significance(self):
        # 100 samples: model A has 80 correct, model B has 65 correct
        pred_a = [True] * 80 + [False] * 20
        pred_b = [True] * 65 + [False] * 35

        chi2, p_val, table = mcnemar_test(pred_a, pred_b)
        self.assertGreater(chi2, 0.0)
        self.assertLess(p_val, 0.05)

    def test_bootstrap_confidence_interval(self):
        data = [78.0, 78.5, 79.0, 78.2, 78.8]
        low, high = compute_bootstrap_ci(data, n_bootstraps=500)
        self.assertLess(low, high)
        self.assertGreaterEqual(low, 77.5)
        self.assertLessEqual(high, 79.5)


if __name__ == "__main__":
    unittest.main()
