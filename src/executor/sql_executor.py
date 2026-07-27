"""
sql_executor.py

Executes generated SQL on the SQLite database.

Responsibilities
----------------
1. Connect to SQLite database
2. Execute SQL
3. Fetch results
4. Return rows
"""

import sqlite3
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:

    success: bool

    rows: list

    columns: list

    error: str = ""


class SQLExecutor:

    def execute(
        self,
        database_path: str,
        sql: str,
    ) -> ExecutionResult:

        try:

            connection = sqlite3.connect(database_path)

            cursor = connection.cursor()

            cursor.execute(sql)

            rows = cursor.fetchall()

            columns = []

            if cursor.description:

                columns = [

                    column[0]

                    for column in cursor.description

                ]

            connection.close()

            return ExecutionResult(

                success=True,

                rows=rows,

                columns=columns,

            )

        except Exception as e:

            return ExecutionResult(

                success=False,

                rows=[],

                columns=[],

                error=str(e),

            )