"""LLM client factory — returns the right client for the configured provider."""

from app.core.config import settings
from app.services.llm_clients.base import BaseLLMClient


def get_llm_client() -> BaseLLMClient:
    """Return the LLM client for the configured provider."""
    provider = settings.llm_provider.lower()

    if provider == "deepseek":
        from app.services.llm_clients.deepseek import DeepSeekClient
        return DeepSeekClient()
    if provider == "openai":
        from app.services.llm_clients.openai import OpenAIClient
        return OpenAIClient()

    # Default to DeepSeek
    from app.services.llm_clients.deepseek import DeepSeekClient
    return DeepSeekClient()
