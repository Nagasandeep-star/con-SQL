"""Reusable orchestration for a single ConSQL inference request with QCE, Oracle join-path, and granular profiling."""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from src.config import (
    JOIN_NECESSITY_THRESHOLD,
    NUM_SERIALIZED_PATHS,
    QCE_MAX_BUDGET,
    QCE_MIN_BUDGET,
)
from src.data_loader import Schema
from src.executor.sql_executor import ExecutionResult, SQLExecutor
from src.graph.graph_builder import SchemaGraph, SchemaGraphBuilder
from src.graph.path_generator import JoinPath, PathGenerationResult, PathGenerator
from src.graph.path_scorer import PathScorer, ScoredPath
from src.llm.sql_generator import SQLGenerator
from src.qce.complexity_estimator import QueryComplexityEstimator
from src.qce.complexity_result import ComplexityResult
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
    complexity_result: Optional[ComplexityResult] = None
    all_selected_paths: List[ScoredPath] = field(default_factory=list)
    latency_breakdown: Dict[str, float] = field(default_factory=dict)


class ConSQLPipeline:
    """Coordinates QCE retrieval, relational graph optimization, generation, validation, and execution."""

    def __init__(
        self,
        linker: Optional[SBERTSchemaLinker] = None,
        graph_builder: Optional[SchemaGraphBuilder] = None,
        path_generator: Optional[PathGenerator] = None,
        path_scorer: Optional[PathScorer] = None,
        sql_generator: Optional[SQLGenerator] = None,
        validator: Optional[SQLValidator] = None,
        executor: Optional[SQLExecutor] = None,
        qce: Optional[QueryComplexityEstimator] = None,
        use_qce: bool = True,
        num_serialized_paths: int = NUM_SERIALIZED_PATHS,
        join_necessity_threshold: float = JOIN_NECESSITY_THRESHOLD,
    ):
        self.linker = linker or SBERTSchemaLinker()
        self.graph_builder = graph_builder or SchemaGraphBuilder()
        self.path_generator = path_generator or PathGenerator()
        self.path_scorer = path_scorer or PathScorer()
        self.sql_generator = sql_generator or SQLGenerator()
        self.validator = validator or SQLValidator()
        self.executor = executor or SQLExecutor()
        self.qce = qce or QueryComplexityEstimator()
        self.use_qce = use_qce
        self.num_serialized_paths = num_serialized_paths
        self.join_necessity_threshold = join_necessity_threshold

    def run(
        self,
        question: str,
        schema: Schema,
        database_path: str,
        oracle_path: Optional[JoinPath] = None,
    ) -> PipelineResult:
        """Run the complete inference pipeline for one question with granular timing."""
        latencies: Dict[str, float] = {}
        t_start = time.perf_counter()

        # ----------------------------------------------------
        # Stage 1: Query Complexity Estimation (QCE)
        # ----------------------------------------------------
        t0 = time.perf_counter()
        complexity = None
        retrieval_budget = None

        if self.use_qce and self.qce is not None:
            try:
                complexity = self.qce.estimate(question)
                # Compute budget: K = K_min + floor(score * (K_max - K_min))
                k_val = QCE_MIN_BUDGET + int(complexity.score * (QCE_MAX_BUDGET - QCE_MIN_BUDGET))
                retrieval_budget = max(QCE_MIN_BUDGET, min(QCE_MAX_BUDGET, k_val))
            except Exception:
                complexity = None
                retrieval_budget = None

        latencies["qce_ms"] = round((time.perf_counter() - t0) * 1000, 2)

        # ----------------------------------------------------
        # Stage 2: Adaptive Schema Retrieval
        # ----------------------------------------------------
        t0 = time.perf_counter()
        try:
            linking = self.linker.link(question, schema, top_k=retrieval_budget)
        except TypeError:
            linking = self.linker.link(question, schema)
        latencies["retrieval_ms"] = round((time.perf_counter() - t0) * 1000, 2)

        # ----------------------------------------------------
        # Stage 3: Relational Graph Construction & Path Generation
        # ----------------------------------------------------
        t0 = time.perf_counter()
        graph = self.graph_builder.build(schema, linking_result=linking)
        paths = self.path_generator.generate(graph, linking, schema)
        latencies["graph_and_paths_ms"] = round((time.perf_counter() - t0) * 1000, 2)

        # ----------------------------------------------------
        # Stage 4: Relational Context Optimization (LRCO / Oracle)
        # ----------------------------------------------------
        t0 = time.perf_counter()
        selected_path: Optional[ScoredPath] = None
        all_selected_paths: List[ScoredPath] = []

        if oracle_path is not None:
            # Oracle Mode: bypass candidate ranking and use ground-truth join path directly
            selected_path = ScoredPath(
                path=oracle_path,
                sem_score=1.0,
                struct_score=1.0,
                length_penalty=0.0,
                final_score=1.0,
                score_breakdown={"oracle": True},
            )
            all_selected_paths = [selected_path]
        elif paths.valid_paths:
            # Check join necessity condition (Eq. 9): only rank if join is needed
            try:
                selection = self.path_scorer.score_and_select(
                    paths, graph, linking, top_k=self.num_serialized_paths
                )
            except TypeError:
                selection = self.path_scorer.score_and_select(
                    paths, graph, linking
                )
            selected_path = selection.best_path
            all_selected_paths = getattr(selection, "top_k_paths", [selected_path])

        latencies["ranking_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        latencies["pre_llm_rco_total_ms"] = round(
            latencies["qce_ms"] + latencies["retrieval_ms"] + latencies["graph_and_paths_ms"] + latencies["ranking_ms"],
            2,
        )

        if not paths.valid_paths and oracle_path is None:
            latencies["total_e2e_ms"] = round((time.perf_counter() - t_start) * 1000, 2)
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
                complexity_result=complexity,
                all_selected_paths=[],
                latency_breakdown=latencies,
            )

        # ----------------------------------------------------
        # Stage 5: Context Serialization & LLM Generation
        # ----------------------------------------------------
        t0 = time.perf_counter()
        tables = selected_path.path.path_table_names if selected_path else [ts.table.name_original for ts in linking.relevant_tables]
        columns = [
            f"{column.table.name_original}.{column.column.name_original}"
            for column in linking.relevant_columns
        ]
        foreign_keys = (
            self._foreign_keys_for_path(schema, selected_path.path)
            if selected_path
            else []
        )

        # If multi-candidate paths are active, serialize top alternatives into join_path context
        serialized_join_paths = tables
        if len(all_selected_paths) > 1:
            multi_path_names = [" -> ".join(sp.path.path_table_names) for sp in all_selected_paths]
            serialized_join_paths = multi_path_names

        generation = self.sql_generator.generate(
            question=question,
            db_id=schema.db_id,
            tables=tables,
            columns=columns,
            join_path=serialized_join_paths,
            foreign_keys=foreign_keys,
        )
        latencies["llm_generation_ms"] = round((time.perf_counter() - t0) * 1000, 2)

        # ----------------------------------------------------
        # Stage 6: SQL Validation & Execution
        # ----------------------------------------------------
        t0 = time.perf_counter()
        validation = self.validator.validate(generation.predicted_sql, schema)
        execution = (
            self.executor.execute(database_path, generation.predicted_sql)
            if validation.is_valid and database_path
            else None
        )
        latencies["validation_and_exec_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        latencies["total_e2e_ms"] = round((time.perf_counter() - t_start) * 1000, 2)

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
            complexity_result=complexity,
            all_selected_paths=all_selected_paths,
            latency_breakdown=latencies,
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
