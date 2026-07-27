"""
base_client.py

Abstract base class for all LLM providers.
"""

from abc import ABC, abstractmethod


class BaseLLMClient(ABC):
    """
    Base interface for all LLM providers.
    """

    @abstractmethod
    def generate_sql(self, prompt: str) -> str:
        """
        Generate SQL from the given prompt.
        """
        pass