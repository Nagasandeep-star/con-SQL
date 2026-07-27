"""Unit tests for evaluator orchestration and stable metric behavior."""

import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from src.config import BIRD_DB_DIR, SPIDER_DB_DIR
from src.evaluator.evaluator import ConSQLEvaluator
from src.executor.sql_executor import ExecutionResult
from src.pipeline import PipelineResult
from src.validator.sql_validator import ValidationResult


class FakePipeline:
    def __init__(self, results):
        self.results = results
        self.calls = []

    def run(self, question, schema, database_path):
        self.calls.append((question, schema, database_path))
        return self.results[question]


def pipeline_result(question, db_id, sql, valid=True, execution_success=True):
    return PipelineResult(
        question=question,
        retrieved_tables=[],
        retrieved_columns=[],
        graph=SimpleNamespace(db_id=db_id),
        candidate_paths=SimpleNamespace(),
        selected_path=None,
        prompt="prompt",
        generated_sql=sql,
        validation_result=ValidationResult(valid, True, valid, False),
        execution_result=ExecutionResult(execution_success, [], []) if valid else None,
    )


class ConSQLEvaluatorTests(unittest.TestCase):
    def setUp(self):
        self.spider_sample = SimpleNamespace(
            question="List users", query="SELECT name FROM users", db_id="spider_db"
        )
        self.bird_sample = SimpleNamespace(
            question="List orders", query="SELECT id FROM orders", db_id="bird_db"
        )
        self.schemas = {
            "spider_db": SimpleNamespace(),
            "bird_db": SimpleNamespace(),
        }
        self.spider_path = os.path.join(SPIDER_DB_DIR, "spider_db", "spider_db.sqlite")
        self.bird_path = os.path.join(BIRD_DB_DIR, "bird_db", "bird_db.sqlite")

    def test_evaluate_uses_pipeline_result_and_keeps_metric_formulas(self):
        pipeline = FakePipeline(
            {
                "List users": pipeline_result(
                    "List users", "spider_db", "select name from users", True, True
                ),
                "List orders": pipeline_result(
                    "List orders", "bird_db", "SELECT other FROM orders", False, False
                ),
            }
        )
        evaluator = ConSQLEvaluator(pipeline=pipeline)

        with patch("src.evaluator.evaluator.os.path.exists", side_effect=lambda path: path in {
            self.spider_path, self.bird_path
        }):
            metrics = evaluator.evaluate([self.spider_sample, self.bird_sample], self.schemas)

        self.assertEqual(2, metrics["Total"])
        self.assertEqual(50.0, metrics["Exact Match (%)"])
        self.assertEqual(50.0, metrics["Valid SQL (%)"])
        self.assertEqual(50.0, metrics["Execution Accuracy (%)"])
        self.assertEqual(2, len(pipeline.calls))
        self.assertEqual(self.spider_path, pipeline.calls[0][2])
        self.assertEqual(self.bird_path, pipeline.calls[1][2])
        self.assertEqual("select name from users", evaluator.results[0].predicted_sql)

    def test_database_path_resolution_supports_spider_and_bird(self):
        evaluator = ConSQLEvaluator(pipeline=FakePipeline({}))

        with patch("src.evaluator.evaluator.os.path.exists", side_effect=lambda path: path in {
            self.spider_path, self.bird_path
        }):
            self.assertEqual(self.spider_path, evaluator.database_path_for("spider_db"))
            self.assertEqual(self.bird_path, evaluator.database_path_for("bird_db"))

    def test_missing_schema_does_not_change_metric_denominator(self):
        evaluator = ConSQLEvaluator(pipeline=FakePipeline({}))
        missing = SimpleNamespace(question="Unknown", query="SELECT 1", db_id="missing")

        metrics = evaluator.evaluate([missing], schemas={})

        self.assertEqual(0, metrics["Total"])
        self.assertEqual(0, metrics["Exact Match (%)"])
        self.assertEqual(0, metrics["Execution Accuracy (%)"])


if __name__ == "__main__":
    unittest.main()
