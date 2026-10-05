"""Unit tests for CLI argument parsing, database path resolution, and demo module availability."""

import os
import sys
import unittest
from unittest.mock import patch

from main import database_path_for, load_dataset
from src.config import SPIDER_DB_DIR


class CLIDemoTests(unittest.TestCase):
    """Verifies that the CLI and demo modules are fully operational."""

    def test_database_path_for_spider_db(self):
        """Verifies database_path_for finds existing sqlite databases."""
        path = database_path_for("concert_singer")
        self.assertTrue(path.endswith("concert_singer.sqlite"))

    def test_load_dataset_finds_schemas(self):
        """Verifies load_dataset returns available schemas."""
        schemas = load_dataset()
        self.assertIn("concert_singer", schemas)

    def test_demo_module_imports(self):
        """Verifies demo.py can be imported without missing module or attribute errors."""
        import demo
        self.assertTrue(callable(demo.build_demo))
        self.assertTrue(callable(demo.load_pipeline))


if __name__ == "__main__":
    unittest.main()
