"""OpenAI LLM Client for ConSQL.

Supports GPT-4o, GPT-4o-mini, and other OpenAI models.
"""

import os
import re
import time
import logging
from typing import Optional

from src.llm.base_client import BaseLLMClient

logger = logging.getLogger(__name__)


class OpenAIClient(BaseLLMClient):
    """Client for generating SQL queries using OpenAI API."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_output_tokens: int = 1024,
    ):
        self.model_name = model_name or os.getenv("OPENAI_MODEL", "gpt-4o")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens

        self._client = None
        self._init_client()

    def _init_client(self):
        """Initialize the OpenAI client."""
        if not self.api_key:
            logger.warning("OPENAI_API_KEY not set. Real API calls will fail.")
            return

        try:
            from openai import OpenAI
            self._client = OpenAI(api_key=self.api_key)
            logger.info("Initialized OpenAI Client for model: %s", self.model_name)
        except Exception as e:
            logger.warning("Failed to initialize OpenAI client: %s", e)

    def _clean_sql(self, text: str) -> str:
        """Strip markdown code fences and extraneous text."""
        text = text.strip()
        pattern = r"```(?:sql)?\s*(.*?)\s*```"
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            text = match.group(1).strip()
        lines = [line.strip() for line in text.splitlines() if line.strip() and not line.strip().startswith("--")]
        return " ".join(lines).strip()

    def generate_sql(self, prompt: str) -> str:
        """Generate SQL using OpenAI Chat Completions API."""
        if self._client is None:
            raise RuntimeError("OpenAI client not initialized. Please set OPENAI_API_KEY.")

        max_retries = 3
        backoff = 2.0

        for attempt in range(max_retries):
            try:
                response = self._client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are an expert SQL engineer. Return only valid SQLite SQL queries with no explanations."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_output_tokens,
                )
                raw_text = response.choices[0].message.content or ""
                return self._clean_sql(raw_text)
            except Exception as e:
                logger.warning("OpenAI API call attempt %d failed: %s", attempt + 1, e)
                if attempt == max_retries - 1:
                    raise
                time.sleep(backoff)
                backoff *= 2.0

        return ""
