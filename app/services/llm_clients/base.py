"""Base LLM client class."""

from typing import Any


class BaseLLMClient:
    """Base class for LLM clients.

    Subclasses implement analyze() to call their specific LLM API.
    """

    async def analyze(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> dict[str, Any]:
        """Send prompts to LLM and return parsed JSON response."""
        raise NotImplementedError
