"""OpenAI LLM client implementation (placeholder).

TODO: Implement OpenAI API client.
Requires OPENAI_API_KEY in environment.
"""

from typing import Any

from app.services.llm_clients.base import BaseLLMClient


class OpenAIClient(BaseLLMClient):
    """OpenAI API client: placeholder implementation."""

    async def analyze(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> dict[str, Any]:
        raise NotImplementedError("OpenAI client not yet implemented")
