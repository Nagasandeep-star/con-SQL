"""Mock LLM Client for offline testing, ablation simulations, and deterministic verification."""

import re
import logging
from src.llm.base_client import BaseLLMClient

logger = logging.getLogger(__name__)


class MockLLMClient(BaseLLMClient):
    """Deterministic mock client that synthesizes syntactically valid SQL from prompt context."""

    def __init__(self, mode: str = "heuristic"):
        self.mode = mode

    def generate_sql(self, prompt: str) -> str:
        """Extract tables and columns from prompt to produce valid SQLite SQL."""
        tables = []
        tables_match = re.search(r"Tables:\s*([^\n]+)", prompt)
        if tables_match and not tables_match.group(1).startswith("None"):
            tables = [t.strip() for t in tables_match.group(1).split(",") if t.strip()]

        # Also support bullet-list format: - singer
        if not tables:
            bullet_matches = re.findall(r"-\s*([a-zA-Z0-9_]+)", prompt)
            if bullet_matches:
                tables = [m for m in bullet_matches if not m.lower() in ("select", "from", "where", "join", "group", "order")]

        columns = []
        columns_match = re.search(r"Columns:\s*([^\n]+)", prompt)
        if columns_match:
            columns = [c.strip() for c in columns_match.group(1).split(",") if c.strip()]

        if not tables:
            tables = ["singer"]

        primary_table = tables[0]

        is_count = bool(re.search(r"\b(how many|count|number of|total)\b", prompt, re.IGNORECASE))
        if is_count:
            select_clause = "count(*)"
        elif columns:
            cleaned_cols = [c.split(".")[-1] for c in columns[:4]]
            select_clause = ", ".join(cleaned_cols)
        else:
            select_clause = "*"

        sql = f"SELECT {select_clause} FROM {primary_table}"

        fk_match = re.findall(r"([a-zA-Z0-9_]+\.[a-zA-Z0-9_]+\s*=\s*[a-zA-Z0-9_]+\.[a-zA-Z0-9_]+)", prompt)
        if len(tables) > 1:
            for other_table in tables[1:3]:
                matching_fk = None
                for fk in fk_match:
                    if other_table in fk and primary_table in fk:
                        matching_fk = fk
                        break
                if matching_fk:
                    sql += f" JOIN {other_table} ON {matching_fk}"
                else:
                    sql += f" NATURAL JOIN {other_table}"

        sql += ";"
        return sql
