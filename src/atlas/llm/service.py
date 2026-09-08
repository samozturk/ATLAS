"""Application-owned chat orchestration and bounded Phase 3 tool calling."""

from collections.abc import AsyncIterator, Sequence

from atlas.config import Settings
from atlas.llm.models import (
    Brain,
    ChatCompletion,
    ChatMessage,
    ChatStreamEvent,
    ConversationMessage,
    MessageRole,
)
from atlas.llm.provider import LLMError, LLMProvider
from atlas.tools import CurrentTimeTool, ToolExecution, ToolRegistry


class ToolRoundLimitError(LLMError):
    """The model requested more tool rounds than ATLAS allows for one turn."""


class ChatService:
    """Adds ATLAS's system prompt while keeping providers free of app policy."""

    def __init__(
        self,
        settings: Settings,
        provider: LLMProvider,
        tool_registry: ToolRegistry | None = None,
    ) -> None:
        self._settings = settings
        self._provider = provider
        self._tool_registry = tool_registry or ToolRegistry(
            [CurrentTimeTool()],
            execution_timeout_seconds=settings.tool_execution_timeout_seconds,
        )

    async def reply(self, conversation: Sequence[ConversationMessage], brain: Brain) -> ChatCompletion:
        messages = self._messages(conversation)
        model = self.model_for(brain)
        for _ in range(self._settings.agent_max_tool_rounds):
            completion = await self._provider.complete(
                messages,
                model=model,
                tools=self._tool_registry.definitions,
            )
            if not completion.message.tool_calls:
                return completion
            await self._append_tool_results(messages, completion.message)
        raise ToolRoundLimitError("ATLAS reached its tool-call limit for this request.")

    async def stream(
        self,
        conversation: Sequence[ConversationMessage],
        brain: Brain,
    ) -> AsyncIterator[ChatStreamEvent]:
        messages = self._messages(conversation)
        model = self.model_for(brain)
        for _ in range(self._settings.agent_max_tool_rounds):
            content: list[str] = []
            tool_calls = []
            done_event: ChatStreamEvent | None = None
            async for event in self._provider.stream(
                messages,
                model=model,
                tools=self._tool_registry.definitions,
            ):
                if event.type == "token":
                    content.append(event.content)
                    yield event.model_copy(update={"brain": brain})
                elif event.type == "tool_call":
                    tool_calls.extend(event.tool_calls)
                    # Keep tool activity visible to API clients without marking
                    # the turn done; a later model pass will form the answer.
                    yield event.model_copy(update={"brain": brain})
                elif event.type == "thinking":
                    yield event.model_copy(update={"brain": brain})
                elif event.type == "done":
                    done_event = event

            if not tool_calls:
                if done_event is None:
                    done_event = ChatStreamEvent(type="done", model=model)
                yield done_event.model_copy(update={"brain": brain})
                return

            tool_results = await self._append_tool_results(
                messages,
                ChatMessage(
                    role=MessageRole.ASSISTANT,
                    content="".join(content),
                    tool_calls=tool_calls,
                ),
            )
            yield ChatStreamEvent(
                type="tool_result",
                tool_results=[result.model_dump(mode="json", exclude_none=True) for result in tool_results],
                brain=brain,
            )
        raise ToolRoundLimitError("ATLAS reached its tool-call limit for this request.")

    def model_for(self, brain: Brain) -> str:
        """Resolve a stable model role without exposing raw tags to clients."""
        if brain is Brain.DEEP:
            return self._settings.deep_model
        return self._settings.fast_model

    def _messages(self, conversation: Sequence[ConversationMessage]) -> list[ChatMessage]:
        return [
            ChatMessage(role=MessageRole.SYSTEM, content=self._settings.llm_system_prompt),
            *(message.as_internal() for message in conversation),
        ]

    async def _append_tool_results(
        self, messages: list[ChatMessage], assistant: ChatMessage
    ) -> list[ToolExecution]:
        messages.append(assistant)
        results = []
        for tool_call in assistant.tool_calls:
            result = await self._tool_registry.execute(tool_call)
            messages.append(result.as_message())
            results.append(result)
        return results
