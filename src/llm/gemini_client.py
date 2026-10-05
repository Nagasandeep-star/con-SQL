"""Gemini LLM Client for ConSQL.

Supports Gemini 2.5 Pro, Gemini 2.5 Flash, and other Google Generative AI models.
Uses google-genai SDK or REST API with automatic retries and exponential backoff.
"""

import os
import re
import time
import logging
from typing import Optional

from src.llm.base_client import BaseLLMClient

logger = logging.getLogger(__name__)


class GeminiClient(BaseLLMClient):
    """Client for generating SQL queries using Google Gemini API."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_output_tokens: int = 1024,
    ):
        self.model_name = model_name or os.getenv("GEMINI_MODEL", "gemini-2.5-pro")
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", os.getenv("GOOGLE_API_KEY", ""))
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens

        self._client = None
        self._init_client()

    def _init_client(self):
        """Initialize the Google GenAI client if API key is available."""
        if not self.api_key:
            logger.warning("GEMINI_API_KEY / GOOGLE_API_KEY not set. Real API calls will fail.")
            return

        try:
            from google import genai
            self._client = genai.Client(api_key=self.api_key)
            logger.info("Initialized google-genai Client successfully for model: %s", self.model_name)
        except Exception as e:
            logger.warning("Failed to initialize google.genai Client: %s. Will fallback to requests.", e)

    def _clean_sql(self, text: str) -> str:
        """Strip markdown code fences and extraneous text from LLM response."""
        text = text.strip()
        # Match ```sql ... ``` or ``` ... ```
        pattern = r"```(?:sql)?\s*(.*?)\s*```"
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            text = match.group(1).strip()
        # If text still contains multiple lines with comments/preambles, extract primary SQL
        lines = [line.strip() for line in text.splitlines() if line.strip() and not line.strip().startswith("--")]
        return " ".join(lines).strip()

    def generate_sql(self, prompt: str) -> str:
        """Generate SQL query with retries and exponential backoff."""
        max_retries = 3
        backoff = 2.0

        for attempt in range(max_retries):
            try:
                if self._client is not None:
                    response = self._client.models.generate_content(
                        model=self.model_name,
                        contents=prompt,
                        config={
                            "temperature": self.temperature,
                            "max_output_tokens": self.max_output_tokens,
                        }
                    )
                    raw_text = response.text or ""
                    return self._clean_sql(raw_text)
                else:
                    # Direct REST API fallback
                    import requests
                    url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent?key={self.api_key}"
                    payload = {
                        "contents": [{"parts": [{"text": prompt}]}],
                        "generationConfig": {
                            "temperature": self.temperature,
                            "maxOutputTokens": self.max_output_tokens,
                        }
                    }
                    res = requests.post(url, json=payload, timeout=60)
                    res.raise_for_status()
                    data = res.json()
                    candidates = data.get("candidates", [])
                    if candidates:
                        raw_text = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "")
                        return self._clean_sql(raw_text)
                    raise RuntimeError(f"Unexpected response structure: {data}")
            except Exception as e:
                logger.warning("Gemini API call attempt %d failed: %s", attempt + 1, e)
                if attempt == max_retries - 1:
                    logger.error("Gemini API failed after %d attempts: %s", max_retries, e)
                    raise
                time.sleep(backoff)
                backoff *= 2.0

        return ""
