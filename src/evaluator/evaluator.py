"""Evaluation orchestration for Spider and BIRD Text-to-SQL datasets."""

import json
import os
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

from src.config import BIRD_DB_DIR, SPIDER_DB_DIR
from src.pipeline import ConSQLPipeline, PipelineResult


@dataclass
class EvaluationResult:
    """Serializable evaluation record compatible with existing metrics tools."""

    question: str
    db_id: str
    gold_sql: str
    predicted_sql: str
    valid_sql: bool
    execution_success: bool
    exact_match: bool
    execution_time: float


class ConSQLEvaluator:
    """Evaluate the current ConSQL pipeline on Spider or BIRD samples."""

    def __init__(
        self,
        pipeline: Optional[ConSQLPipeline] = None,
        database_root: Optional[str] = None,
    ):
        self.pipeline = pipeline or ConSQLPipeline()
        self.database_root = database_root
        self.results: List[EvaluationResult] = []

    @staticmethod
    def normalize(sql: str) -> str:
        return " ".join(sql.lower().split())

    def compute_exact_match(self, predicted: str, gold: str) -> bool:
        return self.normalize(predicted) == self.normalize(gold)

    def evaluate(self, samples, schemas: Dict, max_samples: Optional[int] = None) -> Dict:
        """Run the pipeline for each sample and retain the existing metrics."""
        if max_samples is not None:
            samples = samples[:max_samples]

        total = 0
        exact_matches = 0
        valid_queries = 0
        executable_queries = 0
        self.results = []

        for index, sample in enumerate(samples, start=1):
            print(f"\n[{index}/{len(samples)}] {sample.question}")
            schema = schemas.get(sample.db_id)
            if schema is None:
                print("Schema not found.")
                continue

            database_path = self.database_path_for(sample.db_id)
            if database_path is None:
                print(f"Database not found for db_id: {sample.db_id}")
                continue

            try:
                started_at = time.time()
                pipeline_result = self.pipeline.run(
                    question=sample.question,
                    schema=schema,
                    database_path=database_path,
                )
                elapsed = time.time() - started_at
            except Exception as error:
                print(f"Prediction failed: {error}")
                continue

            result = self._to_evaluation_result(
                pipeline_result=pipeline_result,
                gold_sql=sample.query,
                execution_time=elapsed,
            )
            self.results.append(result)
            total += 1

            if result.exact_match:
                exact_matches += 1
            if result.valid_sql:
                valid_queries += 1
            if result.execution_success:
                executable_queries += 1

        return {
            "Total": total,
            "Exact Match (%)": round(exact_matches / total * 100, 2) if total else 0,
            "Valid SQL (%)": round(valid_queries / total * 100, 2) if total else 0,
            "Execution Accuracy (%)": round(executable_queries / total * 100, 2) if total else 0,
        }

    def database_path_for(self, db_id: str) -> Optional[str]:
        """Resolve a Spider or BIRD SQLite database without dataset-specific prediction code."""
        roots = [self.database_root] if self.database_root else [SPIDER_DB_DIR, BIRD_DB_DIR]
        for root in roots:
            if not root:
                continue
            path = os.path.join(root, db_id, f"{db_id}.sqlite")
            if os.path.exists(path):
                return path
        return None

    def _to_evaluation_result(
        self,
        pipeline_result: PipelineResult,
        gold_sql: str,
        execution_time: float,
    ) -> EvaluationResult:
        """Adapt the current PipelineResult to the stable evaluation record."""
        validation = pipeline_result.validation_result
        execution = pipeline_result.execution_result
        predicted_sql = pipeline_result.generated_sql

        return EvaluationResult(
            question=pipeline_result.question,
            db_id=pipeline_result.graph.db_id,
            gold_sql=gold_sql,
            predicted_sql=predicted_sql,
            valid_sql=validation.is_valid if validation else False,
            execution_success=execution.success if execution else False,
            exact_match=self.compute_exact_match(predicted_sql, gold_sql),
            execution_time=execution_time,
        )

    def save_results(self, output_file: str):
        """Save results in the JSON structure expected by the metrics utility."""
        directory = os.path.dirname(output_file)
        if directory:
            os.makedirs(directory, exist_ok=True)

        with open(output_file, "w", encoding="utf-8") as file:
            json.dump(
                [asdict(result) for result in self.results],
                file,
                indent=4,
                ensure_ascii=False,
            )

        print(f"\nResults saved to {output_file}")

    def print_metrics(self, metrics: Dict):
        print("\n")
        print("=" * 50)
        print("ConSQL Evaluation")
        print("=" * 50)
        for key, value in metrics.items():
            print(f"{key:<25}: {value}")
        print("=" * 50)

    def reset(self):
        """Clear results retained from a previous evaluation run."""
        self.results = []
