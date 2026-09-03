"""Typed contracts shared by LLM providers, the chat API, and future tools."""

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class MessageRole(StrEnum):
    """Roles understood by the LLM conversation contract."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class Brain(StrEnum):
    """Configured model roles; callers cannot choose arbitrary model tags."""

    FAST = "fast"
    DEEP = "deep"


class ToolFunction(BaseModel):
    """A named function and its JSON-compatible arguments."""

    name: str = Field(min_length=1)
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolCall(BaseModel):
    """A future tool request emitted by an LLM, not an authorization to act."""

    id: str | None = None
    type: Literal["function"] = "function"
    function: ToolFunction


class ToolDefinition(BaseModel):
    """Provider-neutral function schema reserved for Phase 3's tool registry."""

    type: Literal["function"] = "function"
    function: dict[str, Any]


class ChatMessage(BaseModel):
    """One message in the internal, provider-neutral conversation format."""

    role: MessageRole
    content: str = ""
    thinking: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_name: str | None = None

    @model_validator(mode="after")
    def validate_tool_message(self) -> "ChatMessage":
        if self.role is MessageRole.TOOL and not self.tool_name:
            raise ValueError("tool messages require tool_name")
        return self

    def as_ollama(self) -> dict[str, Any]:
        """Serialize only fields supported by Ollama's native chat contract."""
        message: dict[str, Any] = {"role": self.role.value, "content": self.content}
        if self.thinking is not None:
            message["thinking"] = self.thinking
        if self.tool_calls:
            message["tool_calls"] = [call.model_dump(mode="json", exclude_none=True) for call in self.tool_calls]
        if self.tool_name is not None:
            message["tool_name"] = self.tool_name
        return message


class Usage(BaseModel):
    """Token counts where the provider exposes them."""

    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)


class ChatCompletion(BaseModel):
    """A complete assistant turn returned by a provider."""

    message: ChatMessage
    model: str
    usage: Usage = Field(default_factory=Usage)


class ChatStreamEvent(BaseModel):
    """A normalized item from an LLM streaming response."""

    type: Literal["thinking", "token", "tool_call", "done"]
    content: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    brain: Brain | None = None
    model: str | None = None
    usage: Usage | None = None


class ConversationMessage(BaseModel):
    """A client-supplied turn; system and tool messages remain server-owned."""

    role: Literal["user", "assistant"]
    content: str = Field(min_length=1)

    def as_internal(self) -> ChatMessage:
        return ChatMessage(role=MessageRole(self.role), content=self.content)


class ChatRequest(BaseModel):
    """A stateless chat request; persistent conversations arrive in Phase 4."""

    messages: list[ConversationMessage] = Field(min_length=1, max_length=100)
    brain: Brain = Brain.FAST

    @model_validator(mode="after")
    def ends_with_user_message(self) -> "ChatRequest":
        if self.messages[-1].role != "user":
            raise ValueError("the final message must have role 'user'")
        return self


class ChatResponse(BaseModel):
    """The public non-streaming chat response."""

    content: str
    brain: Brain
    model: str
    usage: Usage = Field(default_factory=Usage)
