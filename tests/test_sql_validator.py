"""Unit tests for syntax, schema, and join validation."""

import unittest

from src.data_loader import Column, ForeignKey, Schema, Table
from src.validator.sql_validator import SQLValidator


class SQLValidatorTests(unittest.TestCase):
    def setUp(self):
        self.users = Table(0, "users", "users")
        self.orders = Table(1, "orders", "orders")
        self.user_id = Column(1, "id", "id", 0, "number")
        self.user_name = Column(2, "name", "name", 0, "text")
        self.order_id = Column(3, "id", "id", 1, "number")
        self.order_user_id = Column(4, "user_id", "user_id", 1, "number")
        self.users.columns = [self.user_id, self.user_name]
        self.orders.columns = [self.order_id, self.order_user_id]
        self.schema = Schema(
            db_id="test",
            tables=[self.users, self.orders],
            foreign_keys=[ForeignKey(4, 1, 1, 0)],
            columns=[self.user_id, self.user_name, self.order_id, self.order_user_id],
        )
        self.validator = SQLValidator()

    def test_valid_fk_join_is_accepted(self):
        result = self.validator.validate(
            "SELECT users.name FROM users JOIN orders ON users.id = orders.user_id",
            self.schema,
        )

        self.assertTrue(result.is_valid)
        self.assertTrue(result.syntax_valid)
        self.assertTrue(result.schema_valid)
        self.assertEqual([], result.errors)

    def test_unknown_table_and_column_are_reported(self):
        result = self.validator.validate("SELECT missing FROM unknown_table", self.schema)

        self.assertFalse(result.is_valid)
        self.assertIn("Unknown table: unknown_table", result.errors)
        self.assertIn("Unknown column: missing", result.errors)

    def test_join_with_non_fk_equality_is_rejected(self):
        result = self.validator.validate(
            "SELECT users.name FROM users JOIN orders ON users.id = orders.id",
            self.schema,
        )

        self.assertFalse(result.is_valid)
        self.assertIn("foreign-key-consistent", result.error)

    def test_join_without_condition_is_rejected(self):
        result = self.validator.validate(
            "SELECT users.name FROM users JOIN orders",
            self.schema,
        )

        self.assertFalse(result.is_valid)
        self.assertIn("missing an ON or USING condition", result.error)

    def test_invalid_syntax_returns_structured_errors(self):
        result = self.validator.validate("SELECT FROM users", self.schema)

        self.assertFalse(result.is_valid)
        self.assertFalse(result.syntax_valid)
        self.assertEqual(1, len(result.errors))


if __name__ == "__main__":
    unittest.main()
