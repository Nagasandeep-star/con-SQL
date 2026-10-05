"""run_evaluation.py

Runs ConSQL evaluation on Spider/BIRD with cross-backbone support, oracle join-paths,
adaptive QCE budgeting, and granular latency breakdowns.
"""

import argparse
import os
import numpy as np

from src.data_loader import SpiderDataLoader
from src.bird_loader import BIRDMiniDevLoader

from src.evaluator.evaluator import ConSQLEvaluator
from src.evaluator.evaluate_metrics import evaluate
from src.pipeline.service import ConSQLPipeline
from src.llm.sql_generator import SQLGenerator
from src.llm.llm_factory import LLMFactory

from src.config import (
    SPIDER_DIR,
    BIRD_MINIDEV_DIR,
    RESULTS_DIR,
    LLM_PROVIDER,
    LLM_MODEL,
)


def load_dataset(dataset):
    if dataset == "spider":
        loader = SpiderDataLoader(spider_dir=SPIDER_DIR)
        schemas = loader.load_schemas()
        samples = loader.load_dev()
    elif dataset == "bird":
        loader = BIRDMiniDevLoader(bird_minidev_dir=BIRD_MINIDEV_DIR)
        schemas, samples = loader.load()
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    return schemas, samples


def initialize(
    provider: str = LLM_PROVIDER,
    model: str = LLM_MODEL,
    use_qce: bool = True,
    multi_path: int = 1,
):
    """Instantiate pipeline with chosen LLM provider, model, and configuration."""
    llm_client = LLMFactory.create(provider=provider, model=model)
    sql_generator = SQLGenerator(llm_client=llm_client)

    pipeline = ConSQLPipeline(
        sql_generator=sql_generator,
        use_qce=use_qce,
        num_serialized_paths=multi_path,
    )
    return ConSQLEvaluator(pipeline=pipeline)


def run(args):
    print("=" * 65)
    print("ConSQL Comprehensive Benchmark Evaluation")
    print(f"Dataset      : {args.dataset}")
    print(f"LLM Provider : {args.provider}")
    print(f"LLM Model    : {args.model or 'Default'}")
    print(f"Oracle Joins : {args.oracle}")
    print(f"QCE Active   : {not args.no_qce}")
    print(f"Multi-Path K : {args.multi_path}")
    print("=" * 65)

    schemas, samples = load_dataset(args.dataset)

    print(f"\nLoaded {len(schemas)} schemas")
    print(f"Loaded {len(samples)} evaluation samples")

    evaluator = initialize(
        provider=args.provider,
        model=args.model,
        use_qce=not args.no_qce,
        multi_path=args.multi_path,
    )

    metrics = evaluator.evaluate(
        samples=samples,
        schemas=schemas,
        max_samples=args.max_samples,
        use_oracle=args.oracle,
    )

    os.makedirs(RESULTS_DIR, exist_ok=True)

    suffix = f"_{args.provider}_{args.dataset}"
    if args.oracle:
        suffix += "_oracle"
    if args.no_qce:
        suffix += "_no_qce"
    if args.multi_path > 1:
        suffix += f"_multi{args.multi_path}"

    prediction_file = os.path.join(RESULTS_DIR, f"predictions{suffix}.json")
    metrics_file = os.path.join(RESULTS_DIR, f"metrics{suffix}.json")

    evaluator.save_results(prediction_file)

    dataset_root = SPIDER_DIR if args.dataset == "spider" else BIRD_MINIDEV_DIR

    final_metrics = evaluate(
        prediction_file,
        dataset_root,
        metrics_file,
    )

    evaluator.print_metrics(metrics)

    # Compute latency breakdown
    pre_llm_times = []
    e2e_times = []
    for r in evaluator.results:
        bd = getattr(r, "latency_breakdown", {})
        if bd:
            pre_llm_times.append(bd.get("pre_llm_rco_total_ms", 0.0))
            e2e_times.append(bd.get("total_e2e_ms", 0.0))

    if pre_llm_times and e2e_times:
        print("\nLatency Profiling Summary:")
        print(f"  Pre-LLM RCO Latency (Retrieval+Graph+LRCO) : {np.mean(pre_llm_times):.1f} ms")
        print(f"  Total End-to-End Latency (including LLM)  : {np.mean(e2e_times):.1f} ms")

    print("\nEvaluation Completed Successfully.")
    print(f"Predictions saved to: {prediction_file}")
    print(f"Metrics saved to    : {metrics_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ConSQL across benchmarks and LLM backbones.")
    parser.add_argument("--dataset", default="spider", choices=["spider", "bird"], help="Dataset to evaluate")
    parser.add_argument("--provider", default=LLM_PROVIDER, choices=["gemini", "openai", "ollama", "mock"], help="LLM backend provider")
    parser.add_argument("--model", default=LLM_MODEL, help="Specific LLM model identifier")
    parser.add_argument("--oracle", action="store_true", help="Use ground-truth oracle join paths")
    parser.add_argument("--no_qce", action="store_true", help="Disable QCE adaptive schema retrieval")
    parser.add_argument("--multi_path", type=int, default=1, help="Number of serialized join paths (1=Top-1, 2-3=beam/multi-candidate)")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit sample count for rapid verification")

    args = parser.parse_args()
    run(args)