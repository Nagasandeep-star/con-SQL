"""Unit tests for the Ollama LLM client."""

import unittest
from unittest.mock import patch, MagicMock

from src.llm.ollama_client import OllamaClient


class OllamaClientTests(unittest.TestCase):
    def setUp(self):
        self.client = OllamaClient()

    @patch("requests.post")
    def test_generate_sql_success(self, mock_post):
        mock_response = MagicMock()
        mock_response.json.return_value = {"response": "```sql\nSELECT * FROM employees;\n```"}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        sql = self.client.generate_sql("List all employees")
        self.assertEqual("SELECT * FROM employees;", sql)

    def test_clean_sql_markdown_removal(self):
        cleaned = self.client._clean_sql("```sql\nSELECT id FROM t;\n```")
        self.assertEqual("SELECT id FROM t;", cleaned)


if __name__ == "__main__":
    unittest.main()