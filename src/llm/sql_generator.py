"""
sql_generator.py

ConSQL SQL Generator

Coordinates:
- Prompt Builder
- LLM Factory

Supports:
- Gemini
- Groq
- OpenAI

Returns generated SQL.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

from src.llm.base_client import BaseLLMClient
from src.llm.prompt_builder import (
    PromptBuilder,
    PromptContext,
)

from src.llm.llm_factory import LLMFactory


# ----------------------------------------------------
# Result Object
# ----------------------------------------------------

@dataclass
class GenerationResult:
    """
    Output of SQL generation.
    Compatible with the evaluator.
    """

    question: str

    db_id: str

    predicted_sql: str

    gold_sql: str = ""

    join_path_used: list = field(default_factory=list)

    input_prompt: str = ""

    is_valid_sql: bool = True


# ----------------------------------------------------
# SQL Generator
# ----------------------------------------------------

class SQLGenerator:
    """
    Main SQL generation pipeline.

    Responsibilities
    ----------------
    1. Build Prompt
    2. Select LLM Provider
    3. Generate SQL
    4. Return structured result
    """

    def __init__(
        self,
        llm_client: Optional[BaseLLMClient] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
    ):
        self.prompt_builder = PromptBuilder()
        self.llm_client = llm_client or LLMFactory.create(provider=provider, model=model)

    def generate(
        self,
        question: str,
        db_id: str,
        tables: list,
        columns: list,
        join_path: list,
        foreign_keys: list,
        gold_sql: str = "",
    ) -> GenerationResult:

        # ------------------------------------
        # Build Prompt
        # ------------------------------------

        context = PromptContext(

            question=question,

            db_id=db_id,

            tables=tables,

            columns=columns,

            join_path=join_path,

            foreign_keys=foreign_keys,

        )

        prompt = self.prompt_builder.build(context)

        # ------------------------------------
        # Generate SQL
        # ------------------------------------

        sql = self.llm_client.generate_sql(prompt)

        # ------------------------------------
        # Return Result
        # ------------------------------------

        return GenerationResult(

            question=question,

            db_id=db_id,

            predicted_sql=sql,

            gold_sql=gold_sql,

            join_path_used=join_path,

            input_prompt=prompt,

            is_valid_sql=True,

        )