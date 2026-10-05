"""
Ollama Client for ConSQL.
"""

import logging
import re
import time

import requests

from src.config import (
    LLM_MODEL,
    OLLAMA_BASE_URL,
)

from src.llm.base_client import BaseLLMClient

from typing import Optional

logger = logging.getLogger(__name__)


class OllamaClient(BaseLLMClient):

    def __init__(self, model_name: Optional[str] = None, base_url: Optional[str] = None):

        self.model_name = model_name or LLM_MODEL

        self.base_url = base_url or OLLAMA_BASE_URL

        logger.info(
            f"Ollama model loaded: {self.model_name}"
        )

    def generate_sql(self, prompt: str) -> str:

        for attempt in range(3):

            try:

                response = requests.post(

                    f"{self.base_url}/api/generate",

                    json={
                        "model": self.model_name,
                        "prompt": prompt,
                        "stream": False,
                        "keep_alive": "30m",
                        "options": {
                            "temperature": 0,
                            "num_predict": 256,
                        },
                    },
                    timeout=300,

                )

                response.raise_for_status()

                sql = response.json()["response"]

                return self._clean_sql(sql)

            except Exception as e:

                logger.warning(
                    f"Attempt {attempt+1} failed: {e}"
                )

                time.sleep(2)

        raise RuntimeError(
            "Ollama failed after 3 retries."
        )

    @staticmethod
    def _clean_sql(text):

        text = text.strip()

        text = re.sub(
            r"^```sql",
            "",
            text,
            flags=re.IGNORECASE,
        )

        text = re.sub(
            r"^```",
            "",
            text,
        )

        text = re.sub(
            r"```$",
            "",
            text,
        )

        return text.strip()