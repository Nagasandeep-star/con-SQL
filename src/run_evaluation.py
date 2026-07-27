"""
run_evaluation.py

Runs ConSQL evaluation on Spider/BIRD.
"""

import argparse
import os

from src.data_loader import SpiderDataLoader
from src.bird_loader import BIRDMiniDevLoader

from src.evaluator.evaluator import ConSQLEvaluator
from src.evaluator.evaluate_metrics import evaluate

from src.config import (
    SPIDER_DIR,
    BIRD_MINIDEV_DIR,
    RESULTS_DIR,
)


def load_dataset(dataset):

    if dataset == "spider":

        loader = SpiderDataLoader(
            spider_dir=SPIDER_DIR
        )

        schemas = loader.load_schemas()
        samples = loader.load_dev()

    elif dataset == "bird":

        loader = BIRDMiniDevLoader(
            bird_minidev_dir=BIRD_MINIDEV_DIR
        )

        schemas, samples = loader.load()

    else:

        raise ValueError(f"Unknown dataset: {dataset}")

    return schemas, samples


def initialize():

    return ConSQLEvaluator()


def run(dataset):

    print()
    print("=" * 60)
    print("Running ConSQL Evaluation")
    print("=" * 60)

    schemas, samples = load_dataset(dataset)

    print(f"\nLoaded {len(schemas)} schemas")
    print(f"Loaded {len(samples)} evaluation samples")

    evaluator = initialize()

    metrics = evaluator.evaluate(
        samples=samples,
        schemas=schemas,
    )

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True,
    )

    prediction_file = os.path.join(
        RESULTS_DIR,
        f"predictions_{dataset}.json",
    )

    evaluator.save_results(
        prediction_file,
    )
    metrics_file = os.path.join(
        RESULTS_DIR,
        f"metrics_{dataset}.json",
    )

    dataset_root = (
        SPIDER_DIR
        if dataset == "spider"
        else BIRD_MINIDEV_DIR
    )

    final_metrics= evaluate(
        prediction_file,
        dataset_root,
        metrics_file,
    )

    evaluator.print_metrics(
        metrics,
    )

    print("\nEvaluation Completed Successfully.")
    print(f"Predictions saved to: {prediction_file}")
    print(f"Metrics saved to: {metrics_file}")



if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(

        "--dataset",

        default="spider",

        choices=["spider", "bird"],

        help="Dataset to evaluate",

    )

    args = parser.parse_args()

    run(args.dataset)