import os
from typing import Optional

from src.config import LLM_PROVIDER
from src.llm.base_client import BaseLLMClient
from src.llm.ollama_client import OllamaClient
from src.llm.gemini_client import GeminiClient
from src.llm.openai_client import OpenAIClient
from src.llm.mock_client import MockLLMClient


class LLMFactory:
    """Factory for creating LLM clients based on configuration or explicit arguments."""

    @staticmethod
    def create(provider: Optional[str] = None, model: Optional[str] = None) -> BaseLLMClient:
        selected_provider = (provider or LLM_PROVIDER or "gemini").lower()

        if selected_provider in ("gemini", "google"):
            return GeminiClient(model_name=model)

        if selected_provider in ("openai", "gpt"):
            return OpenAIClient(model_name=model)

        if selected_provider == "ollama":
            return OllamaClient(model_name=model)

        if selected_provider in ("mock", "test"):
            return MockLLMClient()

        raise ValueError(
            f"Unsupported LLM provider: '{selected_provider}'. "
            f"Supported providers: ['gemini', 'openai', 'ollama', 'mock']"
        )