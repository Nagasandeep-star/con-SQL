"""Reusable orchestration for a single ConSQL inference request."""

from dataclasses import dataclass
from typing import List, Optional

from src.data_loader import Schema
from src.executor.sql_executor import ExecutionResult, SQLExecutor
from src.graph.graph_builder import SchemaGraph, SchemaGraphBuilder
from src.graph.path_generator import JoinPath, PathGenerationResult, PathGenerator
from src.graph.path_scorer import PathScorer, ScoredPath
from src.llm.sql_generator import SQLGenerator
from src.retrieval.schema_linker import ColumnScore, SBERTSchemaLinker, TableScore
from src.validator.sql_validator import SQLValidator, ValidationResult


@dataclass
class PipelineResult:
    """Structured output from one complete Text-to-SQL inference run."""

    question: str
    retrieved_tables: List[TableScore]
    retrieved_columns: List[ColumnScore]
    graph: SchemaGraph
    candidate_paths: PathGenerationResult
    selected_path: Optional[ScoredPath]
    prompt: str
    generated_sql: str
    validation_result: Optional[ValidationResult]
    execution_result: Optional[ExecutionResult]


class ConSQLPipeline:
    """Coordinates the existing retrieval, generation, validation, and execution stages."""

    def __init__(
        self,
        linker: Optional[SBERTSchemaLinker] = None,
        graph_builder: Optional[SchemaGraphBuilder] = None,
        path_generator: Optional[PathGenerator] = None,
        path_scorer: Optional[PathScorer] = None,
        sql_generator: Optional[SQLGenerator] = None,
        validator: Optional[SQLValidator] = None,
        executor: Optional[SQLExecutor] = None,
    ):
        # Defaults intentionally match the former main.py initialization.
        self.linker = linker or SBERTSchemaLinker()
        self.graph_builder = graph_builder or SchemaGraphBuilder()
        self.path_generator = path_generator or PathGenerator()
        self.path_scorer = path_scorer or PathScorer()
        self.sql_generator = sql_generator or SQLGenerator()
        self.validator = validator or SQLValidator()
        self.executor = executor or SQLExecutor()

    def run(
        self,
        question: str,
        schema: Schema,
        database_path: str,
    ) -> PipelineResult:
        """Run the complete existing inference pipeline for one question."""
        linking = self.linker.link(question, schema)
        graph = self.graph_builder.build(schema, linking_result=linking)
        paths = self.path_generator.generate(graph, linking, schema)

        if not paths.valid_paths:
            return PipelineResult(
                question=question,
                retrieved_tables=linking.relevant_tables,
                retrieved_columns=linking.relevant_columns,
                graph=graph,
                candidate_paths=paths,
                selected_path=None,
                prompt="",
                generated_sql="",
                validation_result=None,
                execution_result=None,
            )

        selection = self.path_scorer.score_and_select(paths, graph, linking)
        selected_path = selection.best_path
        tables = selected_path.path.path_table_names
        columns = [
            f"{column.table.name_original}.{column.column.name_original}"
            for column in linking.relevant_columns
        ]
        foreign_keys = self._foreign_keys_for_path(schema, selected_path.path)

        generation = self.sql_generator.generate(
            question=question,
            db_id=schema.db_id,
            tables=tables,
            columns=columns,
            join_path=tables,
            foreign_keys=foreign_keys,
        )
        validation = self.validator.validate(generation.predicted_sql, schema)
        execution = (
            self.executor.execute(database_path, generation.predicted_sql)
            if validation.is_valid
            else None
        )

        return PipelineResult(
            question=question,
            retrieved_tables=linking.relevant_tables,
            retrieved_columns=linking.relevant_columns,
            graph=graph,
            candidate_paths=paths,
            selected_path=selected_path,
            prompt=generation.input_prompt,
            generated_sql=generation.predicted_sql,
            validation_result=validation,
            execution_result=execution,
        )

    @staticmethod
    def _foreign_keys_for_path(schema: Schema, path: JoinPath) -> List[str]:
        """Format FK constraints for the selected path without changing prompt content."""
        foreign_keys = []
        selected_table_ids = set(path.path_table_ids)

        for fk in schema.foreign_keys:
            if fk.from_table_id not in selected_table_ids or fk.to_table_id not in selected_table_ids:
                continue

            from_column = schema.get_column_by_id(fk.from_col_id)
            to_column = schema.get_column_by_id(fk.to_col_id)
            from_table = schema.get_table_by_id(fk.from_table_id)
            to_table = schema.get_table_by_id(fk.to_table_id)

            if from_column and to_column and from_table and to_table:
                foreign_keys.append(
                    f"{from_table.name_original}.{from_column.name_original}"
                    " = "
                    f"{to_table.name_original}.{to_column.name_original}"
                )

        return foreign_keys


# PipelineService is the application-facing name; ConSQLPipeline remains the
# explicit project name and the backwards-compatible implementation class.
PipelineService = ConSQLPipeline
