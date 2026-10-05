"""
main.py

Unified Command-Line Interface for ConSQL v2.
Supports:
- Interactive query console
- Non-interactive / single-shot batch query execution (--question "..." --db "...")
- Provider selection: gemini, groq, openai, ollama, mock
- Dataset support: Spider and BIRD Mini-Dev
"""

import argparse
import logging
import os
import sys

# Ensure code root and src/ are in sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(CURRENT_DIR, "src")
for p in (CURRENT_DIR, SRC_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from src.config import (
    SPIDER_DB_DIR,
    SPIDER_DIR,
    BIRD_MINIDEV_DIR,
    BIRD_DB_DIR,
    LLM_PROVIDER,
    LLM_MODEL,
)
from src.data_loader import SpiderDataLoader, Schema
from src.bird_loader import BIRDMiniDevLoader
from src.pipeline.service import ConSQLPipeline
from src.llm.sql_generator import SQLGenerator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# -----------------------------------------------------
# Load Datasets
# -----------------------------------------------------

def load_dataset():
    """
    Load schemas from Spider and BIRD Mini-Dev (if available).
    """
    schemas = {}
    if os.path.exists(SPIDER_DIR):
        try:
            spider_schemas = SpiderDataLoader(spider_dir=str(SPIDER_DIR)).load_schemas()
            schemas.update(spider_schemas)
            print(f"Loaded {len(spider_schemas)} Spider schemas")
        except Exception as e:
            logger.warning(f"Could not load Spider schemas: {e}")

    if os.path.exists(BIRD_MINIDEV_DIR):
        try:
            bird_schemas = BIRDMiniDevLoader(bird_minidev_dir=str(BIRD_MINIDEV_DIR)).load_schemas()
            schemas.update(bird_schemas)
            print(f"Loaded {len(bird_schemas)} BIRD schemas")
        except Exception as e:
            logger.warning(f"Could not load BIRD schemas: {e}")

    print(f"Total available schemas: {len(schemas)}")
    return schemas


# -----------------------------------------------------
# Database Path Resolution
# -----------------------------------------------------

def database_path_for(schema_or_db_id) -> str:
    """
    Resolve sqlite path across Spider and BIRD database roots.
    Supports either a Schema object or a database ID string.
    """
    db_id = getattr(schema_or_db_id, "db_id", schema_or_db_id)
    candidates = [
        os.path.join(str(SPIDER_DB_DIR), str(db_id), f"{db_id}.sqlite"),
        os.path.join(str(BIRD_DB_DIR), str(db_id), f"{db_id}.sqlite"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return candidates[0]


# -----------------------------------------------------
# Print Result
# -----------------------------------------------------

def print_result(result):
    print("\n" + "=" * 40)
    print("RETRIEVED TABLES")
    print("=" * 40)
    for table in result.retrieved_tables:
        print(f"- {table.table.name_original} (score: {table.score:.3f})")

    print("\n" + "=" * 40)
    print("RETRIEVED COLUMNS")
    print("=" * 40)
    for col in result.retrieved_columns[:10]:
        print(f"- {col.table.name_original}.{col.column.name_original} (score: {col.score:.3f})")

    print("\n" + "=" * 40)
    print("SELECTED JOIN PATH")
    print("=" * 40)
    if result.selected_path:
        print(" -> ".join(result.selected_path.path.path_table_names))
        print(f"Score: {result.selected_path.final_score:.4f}")
    else:
        print("No join path required / found.")

    print("\n" + "=" * 40)
    print("GENERATED SQL")
    print("=" * 40)
    print(result.generated_sql)

    validation = result.validation_result
    print("\n" + "=" * 40)
    print("VALIDATION")
    print("=" * 40)
    if validation:
        print("SQL Valid :", validation.is_valid)
        if validation.error:
            print("Error     :", validation.error)
    else:
        print("SQL Valid : N/A")

    execution = result.execution_result
    print("\n" + "=" * 40)
    print("EXECUTION")
    print("=" * 40)
    if execution:
        print("Success   :", execution.success)
        if execution.success:
            print(f"Columns   : {', '.join(execution.columns)}")
            print(f"Rows ({len(execution.rows)}):")
            for row in execution.rows[:5]:
                print(f"  {row}")
            if len(execution.rows) > 5:
                print(f"  ... and {len(execution.rows) - 5} more rows")
        else:
            print(f"Error     : {execution.error}")
    else:
        print("Execution : Skipped")

    if result.latency_breakdown:
        print("\n" + "=" * 40)
        print("LATENCY PROFILE")
        print("=" * 40)
        for stage, ms in result.latency_breakdown.items():
            print(f"  {stage:<22}: {ms:.1f} ms")


# -----------------------------------------------------
# Main CLI
# -----------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="ConSQL v2: Context-Aware Semantic Text-to-SQL",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--provider",
        type=str,
        default=LLM_PROVIDER,
        choices=["gemini", "openai", "ollama", "mock"],
        help="LLM Provider to use for SQL generation",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=LLM_MODEL,
        help="Model name identifier",
    )
    parser.add_argument(
        "--db",
        type=str,
        default=None,
        help="Database ID (e.g. concert_singer, financial)",
    )
    parser.add_argument(
        "--question",
        type=str,
        default=None,
        help="Natural language question to execute in single-shot mode",
    )
    parser.add_argument(
        "--no-qce",
        action="store_true",
        help="Disable Query Complexity Estimation (QCE)",
    )

    args = parser.parse_args()

    print("\n" + "=" * 50)
    print("                   ConSQL v2")
    print("      Context-Aware Semantic Text-to-SQL")
    print("=" * 50)
    print(f"LLM Provider : {args.provider}")
    print(f"LLM Model    : {args.model}")

    schemas = load_dataset()
    if not schemas:
        print("[ERROR] No schemas loaded. Run check_setup.py first.")
        sys.exit(1)

    sql_gen = SQLGenerator(provider=args.provider, model=args.model)
    pipeline = ConSQLPipeline(
        sql_generator=sql_gen,
        use_qce=not args.no_qce,
    )

    databases = sorted(list(schemas.keys()))

    # Single-shot execution mode
    if args.question:
        target_db = args.db or databases[0]
        if target_db not in schemas:
            clean_target = target_db.strip().lower().replace(".sqlite", "").replace(".db", "").replace("-", "_").replace(" ", "_")
            matched = None
            for s_id in schemas:
                norm_s = s_id.lower().replace("-", "_").replace(" ", "_")
                if norm_s == clean_target or clean_target in norm_s or norm_s in clean_target:
                    matched = s_id
                    break
            if matched:
                target_db = matched
            else:
                print(f"[ERROR] Database '{target_db}' not found.")
                print(f"Available databases: {', '.join(databases)}")
                sys.exit(1)

        schema = schemas[target_db]
        db_path = database_path_for(target_db)
        print(f"\nTarget Database : {target_db} ({db_path})")
        print(f"Question        : {args.question}")
        res = pipeline.run(args.question, schema, db_path)
        print_result(res)
        return

    # Interactive console mode
    print("\nAvailable Databases:")
    for i, db in enumerate(databases, start=1):
        print(f"  [{i:>2}] {db}")

    try:
        if args.db and args.db in schemas:
            chosen_db = args.db
            print(f"\nPre-selected Database: {chosen_db}")
        else:
            while True:
                choice = input("\nChoose Database [number or name, or 'q' to quit]: ").strip()
                if choice.lower() in ("q", "quit", "exit"):
                    print("Exiting.")
                    return
                if choice.isdigit() and 1 <= int(choice) <= len(databases):
                    chosen_db = databases[int(choice) - 1]
                    break
                elif choice in schemas:
                    chosen_db = choice
                    break
                print("Invalid selection. Please choose a valid index or database ID.")

        schema = schemas[chosen_db]
        db_path = database_path_for(chosen_db)
        print(f"\nActive Database: {chosen_db}")
        print("Enter queries below. Type 'exit' to quit.\n")

        while True:
            question = input("Question > ").strip()
            if not question:
                continue
            if question.lower() in ("exit", "quit", "q"):
                print("Goodbye.")
                break

            try:
                res = pipeline.run(question, schema, db_path)
                print_result(res)
            except Exception as err:
                print(f"[ERROR] Pipeline execution failed: {err}")

    except (EOFError, KeyboardInterrupt):
        print("\nSession ended. Exiting ConSQL CLI.")
        return


if __name__ == "__main__":
    main()