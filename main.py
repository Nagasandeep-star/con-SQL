"""
main.py

Interactive command-line entry point for ConSQL.
Supports:
- Spider
- BIRD
- Gemini
- Groq
- OpenAI
"""

import logging
import os

from src.config import (
    SPIDER_DB_DIR,
    SPIDER_DIR,
    LLM_PROVIDER,
    LLM_MODEL,
)

from src.data_loader import SpiderDataLoader
from src.pipeline import ConSQLPipeline

logging.basicConfig(level=logging.INFO)


# -----------------------------------------------------
# Load Dataset
# -----------------------------------------------------

def load_dataset():
    """
    Load all Spider schemas.
    """

    schemas = SpiderDataLoader(
        spider_dir=SPIDER_DIR
    ).load_schemas()

    print(f"\nLoaded {len(schemas)} schemas")

    return schemas


# -----------------------------------------------------
# Database Path
# -----------------------------------------------------

def database_path_for(schema):

    return os.path.join(
        SPIDER_DB_DIR,
        schema.db_id,
        f"{schema.db_id}.sqlite",
    )


# -----------------------------------------------------
# Print Result
# -----------------------------------------------------

def print_result(result):

    print("\n==============================")
    print("RETRIEVED TABLES")
    print("==============================")

    for table in result.retrieved_tables:
        print("-", table.table.name_original)

    print("\n==============================")
    print("RETRIEVED COLUMNS")
    print("==============================")

    for column in result.retrieved_columns:
        print(
            f"- {column.table.name_original}.{column.column.name_original}"
        )

    if result.selected_path is None:

        print("\nNo valid join path found.")

        return

    print("\n==============================")
    print("SELECTED JOIN PATH")
    print("==============================")

    print(
        " -> ".join(
            result.selected_path.path.path_table_names
        )
    )

    print("\n==============================")
    print("GENERATED SQL")
    print("==============================")

    print(result.generated_sql)

    validation = result.validation_result

    print("\n==============================")
    print("VALIDATION")
    print("==============================")

    print("SQL Valid :", validation.is_valid)

    if validation.error:
        print("Error     :", validation.error)

    if not validation.is_valid:
        return

    execution = result.execution_result

    print("\n==============================")
    print("EXECUTION")
    print("==============================")

    if execution.success:

        print("\nColumns:")

        print(", ".join(execution.columns))

        print("\nRows:")

        if execution.rows:

            for row in execution.rows:
                print(row)

        else:

            print("No rows returned.")

    else:

        print("Execution Error:", execution.error)


# -----------------------------------------------------
# Main
# -----------------------------------------------------

def main():

    print("\n==============================================")
    print("          ConSQL v2")
    print(" Context-Aware Semantic Text-to-SQL")
    print("==============================================")

    print(f"\nLLM Provider : {LLM_PROVIDER}")
    print(f"LLM Model    : {LLM_MODEL}")

    schemas = load_dataset()

    pipeline = ConSQLPipeline()

    print("\nAvailable Databases\n")

    databases = list(schemas)

    for i, db in enumerate(databases, start=1):

        print(f"{i}. {db}")

    while True:

        try:

            index = int(
                input("\nChoose Database : ")
            ) - 1

            if 0 <= index < len(databases):
                break

            print("Invalid database number.")

        except ValueError:

            print("Please enter a valid number.")

    schema = schemas[databases[index]]

    database_path = database_path_for(schema)

    while True:

        question = input(
            "\nQuestion (type exit): "
        )

        if question.lower() == "exit":
            break

        print("\n==============================================")
        print("QUESTION")
        print("==============================================")

        print(question)

        try:

            result = pipeline.run(
                question,
                schema,
                database_path,
            )

            print_result(result)

        except Exception as error:

            print("\nGeneration Error")

            print(error)


if __name__ == "__main__":

    main()