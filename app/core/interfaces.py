"""Protocol interfaces for pluggable components.

Only define interfaces that have multiple implementations.
"""

from typing import Any, Protocol


class ILLMClient(Protocol):
    """LLM client interface — implementations: DeepSeek, OpenAI."""

    async def analyze(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> dict[str, Any]:
        """Send prompts to LLM and return parsed JSON response."""
        ...
