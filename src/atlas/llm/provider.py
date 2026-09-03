"""Minimal provider interface; orchestration and authorization stay in ATLAS."""

from collections.abc import AsyncIterator, Sequence
from typing import Protocol

from atlas.llm.models import ChatCompletion, ChatMessage, ChatStreamEvent, ToolDefinition


class LLMError(Exception):
    """Base exception for an LLM failure safe to surface through the chat API."""


class LLMUnavailableError(LLMError):
    """The configured provider cannot serve the request right now."""


class LLMResponseError(LLMError):
    """The provider responded, but the response was invalid or rejected."""


class LLMProvider(Protocol):
    """Provider boundary used by ATLAS's chat and future agent loop."""

    async def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        model: str,
        tools: Sequence[ToolDefinition] = (),
    ) -> ChatCompletion:
        """Return one complete assistant turn."""

    def stream(
        self,
        messages: Sequence[ChatMessage],
        *,
        model: str,
        tools: Sequence[ToolDefinition] = (),
    ) -> AsyncIterator[ChatStreamEvent]:
        """Yield a normalized assistant turn as it arrives."""

    async def aclose(self) -> None:
        """Release provider-owned network resources."""
