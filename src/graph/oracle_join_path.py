"""Oracle Join Path Extractor for ConSQL.

Extracts ground-truth join paths directly from reference/gold SQL queries.
Enables the Oracle Join-Path Experiment requested by reviewers to decouple
LRCO ranking errors from downstream LLM reasoning limitations.
"""

from typing import List, Optional, Set
import sqlglot
from sqlglot import exp

from src.data_loader import Schema
from src.graph.path_generator import JoinPath


class OracleJoinPathExtractor:
    """Extracts ground-truth join path from gold SQL and database schema."""

    @staticmethod
    def extract_tables(gold_sql: str) -> List[str]:
        """Extract all table names referenced in the gold SQL query."""
        if not gold_sql or not gold_sql.strip():
            return []

        tables: Set[str] = set()
        try:
            parsed = sqlglot.parse_one(gold_sql, read="sqlite")
            for table_expr in parsed.find_all(exp.Table):
                if table_expr.name:
                    tables.add(table_expr.name.lower())
        except Exception:
            # Fallback simple regex extraction
            import re
            tokens = re.findall(r"\bFROM\s+([a-zA-Z0-9_]+)|\bJOIN\s+([a-zA-Z0-9_]+)", gold_sql, re.IGNORECASE)
            for t1, t2 in tokens:
                if t1:
                    tables.add(t1.lower())
                if t2:
                    tables.add(t2.lower())

        return sorted(list(tables))

    @classmethod
    def get_oracle_join_path(cls, gold_sql: str, schema: Schema) -> Optional[JoinPath]:
        """Construct a JoinPath object representing the ground truth tables from gold SQL."""
        table_names = cls.extract_tables(gold_sql)
        if not table_names:
            return None

        # Map table names to schema table IDs and original names
        table_ids: List[int] = []
        resolved_names: List[str] = []

        for name in table_names:
            t = schema.get_table_by_name(name)
            if t is not None:
                table_ids.append(t.table_id)
                resolved_names.append(t.name_original)

        if not table_ids:
            return None

        # Build JoinPath
        path = JoinPath(path_table_ids=table_ids, path_table_names=resolved_names)
        return path
