"""
prompt_builder.py

Builds the prompt supplied to the LLM.

The prompt contains:
- User question
- Database information
- Relevant schema
- Join path
- Foreign keys
- SQL generation rules
"""

from dataclasses import dataclass
from typing import List


# ----------------------------------------------------
# Prompt Context
# ----------------------------------------------------

@dataclass
class PromptContext:
    question: str
    db_id: str

    tables: List[str]

    columns: List[str]

    join_path: List[str]

    foreign_keys: List[str]


# ----------------------------------------------------
# Prompt Builder
# ----------------------------------------------------

class PromptBuilder:

    def _format_tables(self, tables: List[str]) -> str:

        if not tables:
            return "None"

        return "\n".join(f"- {table}" for table in tables)

    def _format_columns(self, columns: List[str]) -> str:

        if not columns:
            return "None"

        return "\n".join(f"- {column}" for column in columns)

    def _format_join_path(self, join_path: List[str]) -> str:

        if not join_path:
            return "No join required."

        return " -> ".join(join_path)

    def _format_foreign_keys(self, foreign_keys: List[str]) -> str:

        if not foreign_keys:
            return "None"

        return "\n".join(f"- {fk}" for fk in foreign_keys)

    def build(self, context: PromptContext) -> str:

        tables = self._format_tables(context.tables)

        columns = self._format_columns(context.columns)

        join_path = self._format_join_path(context.join_path)

        foreign_keys = self._format_foreign_keys(context.foreign_keys)

        prompt = f"""
You are an expert SQLite SQL generation assistant.

Your task is to convert a natural language question into a valid SQLite SQL query.

Return ONLY the SQL query.

Do NOT explain your reasoning.

Do NOT use markdown.

Do NOT wrap the SQL inside code fences.

==================================================
DATABASE
==================================================

Database ID:
{context.db_id}

==================================================
QUESTION
==================================================

{context.question}

==================================================
RELEVANT TABLES
==================================================

{tables}

==================================================
RELEVANT COLUMNS
==================================================

{columns}

==================================================
JOIN PATH
==================================================

{join_path}

==================================================
FOREIGN KEY RELATIONSHIPS
==================================================

{foreign_keys}

==================================================
SQL GENERATION RULES
==================================================

1. Use SQLite syntax.

2. Use ONLY the provided tables.

3. Use ONLY the provided columns.

4. Follow the supplied join path whenever joins are required.

5. Respect all foreign key relationships.

6. Do NOT invent tables.

7. Do NOT invent columns.

8. Do NOT use SELECT * unless absolutely necessary.

9. Use aggregate functions only when required by the question.

10. Produce a syntactically correct SQL query.

11. Return ONLY the SQL query.

==================================================
SQL
==================================================
"""

        return prompt.strip()