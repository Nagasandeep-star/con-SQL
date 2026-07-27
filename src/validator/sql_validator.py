"""
sql_validator.py

ConSQL SQL Validator

Responsibilities
----------------
1. SQL Syntax Validation
2. Lightweight Schema Validation
3. Returns structured validation results

NOTE:
Actual SQL execution is performed by SQLExecutor.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import sqlglot
from sqlglot import exp

from src.data_loader import Schema, Table


# ---------------------------------------------------------
# Validation Result
# ---------------------------------------------------------

@dataclass
class ValidationResult:
    """
    Result returned after validating a SQL query.
    """

    is_valid: bool

    syntax_valid: bool

    schema_valid: bool

    execution_valid: bool

    error: Optional[str] = None

    errors: List[str] = field(default_factory=list)


# ---------------------------------------------------------
# SQL Validator
# ---------------------------------------------------------

class SQLValidator:
    """
    Performs lightweight validation before execution.

    Responsibilities
    ----------------
    ✓ SQL syntax validation
    ✓ Optional schema validation

    Does NOT execute SQL.
    """

    def __init__(self):
        pass

    # -----------------------------------------------------
    # Syntax Validation
    # -----------------------------------------------------

    def validate_syntax(self, sql: str) -> bool:
        """
        Validate SQL syntax using sqlglot.
        """

        if not sql:
            return False

        try:
            expression = sqlglot.parse_one(sql, read="sqlite")
            return self._syntax_error(expression) is None

        except Exception:

            return False

    # -----------------------------------------------------
    # Schema Validation
    # -----------------------------------------------------

    def validate_schema(self, sql: str, schema: Optional[Schema] = None) -> bool:
        """Check that the query's table and column references fit ``schema``.

        A schema is optional to retain the validator's syntax-only use case.
        When supplied, table names, aliases, and column references are checked
        without opening or executing against a database.
        """
        if schema is None:
            return True

        try:
            expression = sqlglot.parse_one(sql, read="sqlite")
        except Exception:
            return False

        return self._syntax_error(expression) is None and not self._schema_errors(expression, schema)

    def validate(self, sql: str, schema: Optional[Schema] = None) -> ValidationResult:
        """Validate SQL syntax and its basic consistency with a schema.

        Validation is deliberately non-executing. It verifies that referenced
        source tables exist, qualified columns resolve through a table/alias,
        and unqualified columns occur in exactly one query source table.
        """
        if not sql:
            return ValidationResult(
                is_valid=False,
                syntax_valid=False,
                schema_valid=False,
                execution_valid=False,
                error="SQL is empty.",
                errors=["SQL is empty."],
            )

        try:
            expression = sqlglot.parse_one(sql, read="sqlite")
        except Exception as error:
            return ValidationResult(
                is_valid=False,
                syntax_valid=False,
                schema_valid=False,
                execution_valid=False,
                error=f"Invalid SQLite SQL syntax: {error}",
                errors=[f"Invalid SQLite SQL syntax: {error}"],
            )

        syntax_error = self._syntax_error(expression)
        if syntax_error:
            return ValidationResult(
                is_valid=False,
                syntax_valid=False,
                schema_valid=False,
                execution_valid=False,
                error=syntax_error,
                errors=[syntax_error],
            )

        errors = self._schema_errors(expression, schema) if schema else []
        return ValidationResult(
            is_valid=not errors,
            syntax_valid=True,
            schema_valid=not errors,
            execution_valid=False,
            error="; ".join(errors) if errors else None,
            errors=errors,
        )

    @staticmethod
    def _syntax_error(expression: exp.Expression) -> Optional[str]:
        """Catch malformed forms that sqlglot can recover into an AST."""
        for select in expression.find_all(exp.Select):
            if not select.expressions:
                return "Invalid SQLite SQL syntax: SELECT requires at least one expression."
        return None

    def _schema_errors(self, expression: exp.Expression, schema: Schema) -> List[str]:
        """Return basic table/column consistency errors for a parsed query."""
        tables_by_name = self._tables_by_name(schema)
        cte_names = {
            cte.alias_or_name.lower()
            for cte in expression.find_all(exp.CTE)
            if cte.alias_or_name
        }
        source_tables: Dict[str, Table] = {}
        errors: List[str] = []

        for table_ref in expression.find_all(exp.Table):
            table_name = table_ref.name.lower()

            # CTEs are query-local derived sources, rather than schema tables.
            if table_name in cte_names:
                continue

            table = tables_by_name.get(table_name)
            if table is None:
                errors.append(f"Unknown table: {table_ref.name}")
                continue

            source_tables[table_ref.alias_or_name.lower()] = table
            source_tables[table_name] = table

        for column_ref in expression.find_all(exp.Column):
            column_name = column_ref.name
            if not column_name or column_name == "*":
                continue

            qualifier = column_ref.table
            if qualifier:
                source = source_tables.get(qualifier.lower())
                if source is None:
                    # A CTE/derived-table column cannot be checked against the
                    # physical Schema model, so leave it to the database.
                    if qualifier.lower() not in cte_names:
                        errors.append(f"Unknown table or alias '{qualifier}' for column '{column_name}'")
                    continue

                if not self._table_has_column(source, column_name):
                    errors.append(f"Unknown column '{column_name}' in table '{source.name_original}'")
                continue

            unique_sources = {
                table.table_id: table
                for table in source_tables.values()
            }.values()
            matches = [
                table for table in unique_sources
                if self._table_has_column(table, column_name)
            ]
            if not matches:
                errors.append(f"Unknown column: {column_name}")
            elif len(matches) > 1:
                errors.append(f"Ambiguous unqualified column: {column_name}")

        errors.extend(self._join_errors(expression, schema, source_tables))

        # Preserve error order while avoiding repeated messages from expressions.
        return list(dict.fromkeys(errors))

    def _join_errors(
        self,
        expression: exp.Expression,
        schema: Schema,
        source_tables: Dict[str, Table],
    ) -> List[str]:
        """Validate basic JOIN predicates against declared foreign keys.

        Non-CROSS joins must use ``ON`` or ``USING``. For ``ON`` joins, at
        least one equality must connect the joined table to another source via
        a declared schema foreign key. ``USING`` columns must exist on the
        joined source and participate in a matching FK relation.
        """
        errors: List[str] = []

        for join in expression.find_all(exp.Join):
            if join.args.get("kind") == "CROSS":
                continue

            joined_table_ref = join.this
            if not isinstance(joined_table_ref, exp.Table):
                # Derived tables cannot be checked against the physical schema.
                continue

            joined_alias = joined_table_ref.alias_or_name.lower()
            joined_table = source_tables.get(joined_alias)
            if joined_table is None:
                continue

            using_columns = join.args.get("using") or []
            on_condition = join.args.get("on")
            if using_columns:
                errors.extend(
                    self._using_join_errors(joined_table, using_columns, schema, source_tables)
                )
                continue

            # sqlglot represents a missing ON clause as ON TRUE for standard joins.
            if on_condition is None or isinstance(on_condition, exp.Boolean):
                errors.append(f"JOIN for table '{joined_table.name_original}' is missing an ON or USING condition")
                continue

            fk_match_found = False
            for equality in on_condition.find_all(exp.EQ):
                left, right = equality.this, equality.expression
                if not isinstance(left, exp.Column) or not isinstance(right, exp.Column):
                    continue

                left_table = source_tables.get(left.table.lower()) if left.table else None
                right_table = source_tables.get(right.table.lower()) if right.table else None
                if left_table is None or right_table is None:
                    continue
                if joined_table not in (left_table, right_table):
                    continue

                if self._columns_form_foreign_key(schema, left_table, left.name, right_table, right.name):
                    fk_match_found = True
                    break

            if not fk_match_found:
                errors.append(
                    f"JOIN for table '{joined_table.name_original}' has no foreign-key-consistent equality condition"
                )

        return errors

    def _using_join_errors(
        self,
        joined_table: Table,
        using_columns: List[exp.Identifier],
        schema: Schema,
        source_tables: Dict[str, Table],
    ) -> List[str]:
        """Validate ``USING`` columns for existence and FK consistency."""
        errors: List[str] = []
        other_tables = {
            table.table_id: table
            for table in source_tables.values()
            if table.table_id != joined_table.table_id
        }.values()

        for identifier in using_columns:
            column_name = identifier.name
            if not self._table_has_column(joined_table, column_name):
                errors.append(f"Unknown USING column '{column_name}' in table '{joined_table.name_original}'")
                continue

            fk_match_found = any(
                self._table_has_column(other_table, column_name)
                and self._columns_form_foreign_key(
                    schema, joined_table, column_name, other_table, column_name
                )
                for other_table in other_tables
            )
            if not fk_match_found:
                errors.append(
                    f"USING column '{column_name}' in table '{joined_table.name_original}' "
                    "does not match a declared foreign key"
                )

        return errors

    def _columns_form_foreign_key(
        self,
        schema: Schema,
        left_table: Table,
        left_column_name: str,
        right_table: Table,
        right_column_name: str,
    ) -> bool:
        """Return whether two named columns form a declared FK pair in either direction."""
        left_column = self._find_column(left_table, left_column_name)
        right_column = self._find_column(right_table, right_column_name)
        if left_column is None or right_column is None:
            return False

        return any(
            (fk.from_col_id == left_column.col_id and fk.to_col_id == right_column.col_id)
            or (fk.from_col_id == right_column.col_id and fk.to_col_id == left_column.col_id)
            for fk in schema.foreign_keys
        )

    @staticmethod
    def _tables_by_name(schema: Schema) -> Dict[str, Table]:
        tables = {}
        for table in schema.tables:
            tables[table.name.lower()] = table
            tables[table.name_original.lower()] = table
        return tables

    @staticmethod
    def _table_has_column(table: Table, column_name: str) -> bool:
        return SQLValidator._find_column(table, column_name) is not None

    @staticmethod
    def _find_column(table: Table, column_name: str):
        normalized = column_name.lower()
        return next((
            column for column in table.columns
            if column.name.lower() == normalized
            or column.name_original.lower() == normalized
        ), None)
   
