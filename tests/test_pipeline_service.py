"""Unit tests for pipeline orchestration with dependency-injected components."""

from types import SimpleNamespace
import unittest

from src.data_loader import Column, Schema, Table
from src.executor.sql_executor import ExecutionResult
from src.graph.path_generator import JoinPath
from src.pipeline import PipelineService
from src.validator.sql_validator import ValidationResult


class FakeLinker:
    def __init__(self, linking_result):
        self.linking_result = linking_result
        self.calls = []

    def link(self, question, schema):
        self.calls.append((question, schema))
        return self.linking_result


class FakeGraphBuilder:
    def __init__(self, graph):
        self.graph = graph

    def build(self, schema, linking_result):
        return self.graph


class FakePathGenerator:
    def __init__(self, result):
        self.result = result

    def generate(self, graph, linking_result, schema):
        return self.result


class FakePathScorer:
    def __init__(self, selected_path):
        self.selected_path = selected_path

    def score_and_select(self, paths, graph, linking):
        return SimpleNamespace(best_path=self.selected_path)


class FakeSQLGenerator:
    def __init__(self):
        self.arguments = None

    def generate(self, **kwargs):
        self.arguments = kwargs
        return SimpleNamespace(
            predicted_sql="SELECT users.name FROM users",
            input_prompt="test prompt",
        )


class FakeValidator:
    def __init__(self, result):
        self.result = result
        self.arguments = None

    def validate(self, sql, schema):
        self.arguments = (sql, schema)
        return self.result


class FakeExecutor:
    def __init__(self):
        self.arguments = None

    def execute(self, database_path, sql):
        self.arguments = (database_path, sql)
        return ExecutionResult(True, [("Ada",)], ["name"])


class PipelineServiceTests(unittest.TestCase):
    def setUp(self):
        self.users = Table(table_id=0, name="users", name_original="users")
        self.name = Column(1, "name", "name", 0, "text")
        self.users.columns = [self.name]
        self.schema = Schema("example", [self.users], [], [self.name])

        self.linking = SimpleNamespace(
            relevant_tables=[SimpleNamespace(table=self.users)],
            relevant_columns=[SimpleNamespace(table=self.users, column=self.name)],
            get_table_ids=lambda: [0],
        )
        self.graph = SimpleNamespace(db_id="example")
        self.path = JoinPath([0], ["users"])
        self.paths = SimpleNamespace(valid_paths=[self.path])
        self.selected = SimpleNamespace(path=self.path)

    def build_pipeline(self, validation):
        self.generator = FakeSQLGenerator()
        self.executor = FakeExecutor()
        return PipelineService(
            linker=FakeLinker(self.linking),
            graph_builder=FakeGraphBuilder(self.graph),
            path_generator=FakePathGenerator(self.paths),
            path_scorer=FakePathScorer(self.selected),
            sql_generator=self.generator,
            validator=FakeValidator(validation),
            executor=self.executor,
        )

    def test_run_returns_all_pipeline_outputs_and_executes_valid_sql(self):
        pipeline = self.build_pipeline(
            ValidationResult(True, True, True, False)
        )

        result = pipeline.run("List user names", self.schema, "example.sqlite")

        self.assertEqual("List user names", result.question)
        self.assertEqual(self.linking.relevant_tables, result.retrieved_tables)
        self.assertEqual(self.linking.relevant_columns, result.retrieved_columns)
        self.assertIs(self.graph, result.graph)
        self.assertIs(self.paths, result.candidate_paths)
        self.assertIs(self.selected, result.selected_path)
        self.assertEqual("test prompt", result.prompt)
        self.assertEqual("SELECT users.name FROM users", result.generated_sql)
        self.assertTrue(result.validation_result.is_valid)
        self.assertTrue(result.execution_result.success)
        self.assertEqual(("example.sqlite", result.generated_sql), self.executor.arguments)
        self.assertEqual(["users"], self.generator.arguments["tables"])
        self.assertEqual(["users.name"], self.generator.arguments["columns"])

    def test_run_skips_execution_when_validation_fails(self):
        pipeline = self.build_pipeline(
            ValidationResult(False, True, False, False, "Unknown column")
        )

        result = pipeline.run("List user names", self.schema, "example.sqlite")

        self.assertFalse(result.validation_result.is_valid)
        self.assertIsNone(result.execution_result)
        self.assertIsNone(self.executor.arguments)

    def test_run_returns_partial_result_when_no_join_path_exists(self):
        self.paths = SimpleNamespace(valid_paths=[])
        pipeline = self.build_pipeline(
            ValidationResult(True, True, True, False)
        )

        result = pipeline.run("List user names", self.schema, "example.sqlite")

        self.assertIsNone(result.selected_path)
        self.assertEqual("", result.prompt)
        self.assertEqual("", result.generated_sql)
        self.assertIsNone(result.validation_result)
        self.assertIsNone(result.execution_result)


if __name__ == "__main__":
    unittest.main()
