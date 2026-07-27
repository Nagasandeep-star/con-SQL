"""
predict.py

Batch prediction utility for ConSQL.

Runs the complete ConSQL pipeline over Spider or BIRD
and saves predictions for later evaluation.
"""

import argparse
import json
import os
from dataclasses import asdict

from src.config import (
    SPIDER_DIR,
    SPIDER_DB_DIR,
    BIRD_DIR,
    BIRD_DB_DIR,
)

from src.data_loader import (
    SpiderDataLoader,
    BirdDataLoader,
)

from src.pipeline import ConSQLPipeline
from src.evaluator.evaluator import EvaluationResult


def database_path(root, db_id):

    return os.path.join(
        root,
        db_id,
        f"{db_id}.sqlite",
    )


def load_dataset(dataset):

    if dataset.lower() == "spider":

        loader = SpiderDataLoader(
            spider_dir=SPIDER_DIR
        )

        schemas = loader.load_schemas()

        samples = loader.load_dev()

        db_root = SPIDER_DB_DIR

    elif dataset.lower() == "bird":

        loader = BirdDataLoader(
            bird_dir=BIRD_DIR
        )

        schemas = loader.load_schemas()

        samples = loader.load_dev()

        db_root = BIRD_DB_DIR

    else:

        raise ValueError("Unknown dataset.")

    return samples, schemas, db_root


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        choices=["spider", "bird"],
        required=True,
    )

    parser.add_argument(
        "--output",
        required=True,
    )

    args = parser.parse_args()

    samples, schemas, db_root = load_dataset(
        args.dataset
    )

    pipeline = ConSQLPipeline()

    predictions = []

    total = len(samples)

    for i, sample in enumerate(samples, start=1):

        print(f"[{i}/{total}] {sample.question}")

        schema = schemas[sample.db_id]

        db_path = database_path(
            db_root,
            sample.db_id,
        )

        try:

            result = pipeline.run(
                question=sample.question,
                schema=schema,
                database_path=db_path,
            )

            predictions.append(
                EvaluationResult(
                    question=sample.question,
                    db_id=sample.db_id,
                    gold_sql=sample.query,
                    predicted_sql=result.generated_sql,
                    valid_sql=result.validation_result.is_valid,
                    execution_success=(
                        result.execution_result.success
                        if result.execution_result
                        else False
                    ),
                    exact_match=False,
                    execution_time=0.0,
                )
            )

        except Exception as e:

            print(e)

    os.makedirs(
        os.path.dirname(args.output),
        exist_ok=True,
    )

    with open(
        args.output,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            [asdict(p) for p in predictions],
            f,
            indent=4,
            ensure_ascii=False,
        )

    print()

    print("=" * 60)

    print(f"Saved {len(predictions)} predictions.")

    print("=" * 60)


if __name__ == "__main__":

    main()