"""Provider-neutral LLM interfaces and the Ollama implementation."""

from atlas.llm.models import Brain, ChatCompletion, ChatMessage, ChatStreamEvent, MessageRole
from atlas.llm.ollama import OllamaProvider
from atlas.llm.provider import LLMError, LLMProvider, LLMResponseError, LLMUnavailableError
from atlas.llm.service import ChatService

__all__ = [
    "ChatCompletion",
    "ChatMessage",
    "ChatService",
    "ChatStreamEvent",
    "Brain",
    "LLMError",
    "LLMProvider",
    "LLMResponseError",
    "LLMUnavailableError",
    "MessageRole",
    "OllamaProvider",
]
