from src.config import LLM_PROVIDER

from src.llm.ollama_client import OllamaClient


class LLMFactory:

    @staticmethod
    def create():

        provider = LLM_PROVIDER.lower()

        if provider == "ollama":
            return OllamaClient()

        raise ValueError(
            f"Unsupported LLM provider: {provider}"
        )