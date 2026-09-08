import asyncio
from collections.abc import AsyncIterator, Sequence

from fastapi.testclient import TestClient

from atlas.config import Settings
from atlas.llm.models import (
    Brain,
    ChatCompletion,
    ChatMessage,
    ChatStreamEvent,
    ConversationMessage,
    MessageRole,
    ToolCall,
    ToolDefinition,
    ToolFunction,
)
from atlas.main import create_app
from atlas.llm.service import ChatService


class FakeProvider:
    """In-memory provider used to test the HTTP and orchestration boundary."""

    def __init__(self) -> None:
        self.messages: list[ChatMessage] = []
        self.model: str | None = None
        self.closed = False
        self.tools: list[ToolDefinition] = []

    async def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        model: str,
        tools: Sequence[ToolDefinition] = (),
    ) -> ChatCompletion:
        self.messages = list(messages)
        self.model = model
        self.tools = list(tools)
        return ChatCompletion(
            message=ChatMessage(role=MessageRole.ASSISTANT, content="Hello from the test provider."),
            model="test-model",
        )

    async def stream(
        self,
        messages: Sequence[ChatMessage],
        *,
        model: str,
        tools: Sequence[ToolDefinition] = (),
    ) -> AsyncIterator[ChatStreamEvent]:
        self.messages = list(messages)
        self.model = model
        self.tools = list(tools)
        yield ChatStreamEvent(type="token", content="Hello")
        yield ChatStreamEvent(type="token", content=" there")
        yield ChatStreamEvent(type="done", model="test-model")

    async def aclose(self) -> None:
        self.closed = True


def test_chat_adds_system_prompt_and_returns_provider_reply() -> None:
    provider = FakeProvider()
    app = create_app(Settings(environment="test", llm_system_prompt="System rule."), provider=provider)

    with TestClient(app) as client:
        response = client.post("/chat", json={"messages": [{"role": "user", "content": "Hello"}]})

    assert response.status_code == 200
    assert response.json() == {
        "content": "Hello from the test provider.",
        "brain": "fast",
        "model": "test-model",
        "usage": {"prompt_tokens": None, "completion_tokens": None},
    }
    assert provider.messages == [
        ChatMessage(role=MessageRole.SYSTEM, content="System rule."),
        ChatMessage(role=MessageRole.USER, content="Hello"),
    ]
    assert provider.model == "qwen3.6:35b"
    assert provider.tools[0].function["name"] == "get_current_time"
    assert provider.closed is True


def test_chat_requires_a_final_user_message() -> None:
    app = create_app(Settings(environment="test"), provider=FakeProvider())

    with TestClient(app) as client:
        response = client.post("/chat", json={"messages": [{"role": "assistant", "content": "Hello"}]})

    assert response.status_code == 422
    assert "final message" in response.text


def test_streaming_chat_returns_server_sent_events() -> None:
    app = create_app(Settings(environment="test"), provider=FakeProvider())

    with TestClient(app) as client:
        response = client.post(
            "/chat/stream",
            json={"messages": [{"role": "user", "content": "Hello"}]},
        )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert 'event: token\ndata: {"type":"token","content":"Hello","tool_calls":[],"tool_results":[],"brain":"fast"}' in response.text
    assert 'event: done\ndata: {"type":"done","content":"","tool_calls":[],"tool_results":[],"brain":"fast","model":"test-model"}' in response.text


class ToolCallingProvider:
    """Returns a time request first, then a final response after the result."""

    def __init__(self) -> None:
        self.requests: list[list[ChatMessage]] = []

    async def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        model: str,
        tools: Sequence[ToolDefinition] = (),
    ) -> ChatCompletion:
        assert model == "qwen3.6:35b"
        assert tools[0].function["name"] == "get_current_time"
        self.requests.append(list(messages))
        if len(self.requests) == 1:
            return ChatCompletion(
                message=ChatMessage(
                    role=MessageRole.ASSISTANT,
                    tool_calls=[
                        ToolCall(
                            id="time-1",
                            function=ToolFunction(name="get_current_time", arguments={}),
                        )
                    ],
                ),
                model=model,
            )
        return ChatCompletion(
            message=ChatMessage(role=MessageRole.ASSISTANT, content="It is time to test tools."),
            model=model,
        )

    async def stream(
        self,
        messages: Sequence[ChatMessage],
        *,
        model: str,
        tools: Sequence[ToolDefinition] = (),
    ) -> AsyncIterator[ChatStreamEvent]:
        raise AssertionError("stream is not used by this test")
        yield ChatStreamEvent(type="done")

    async def aclose(self) -> None:
        return None


def test_chat_service_executes_a_registered_tool_before_finishing() -> None:
    provider = ToolCallingProvider()
    service = ChatService(Settings(environment="test", llm_system_prompt="System rule."), provider)

    async def exercise() -> ChatCompletion:
        return await service.reply(
            [ConversationMessage(role="user", content="What time is it?")],
            Brain.FAST,
        )

    response = asyncio.run(exercise())

    assert response.message.content == "It is time to test tools."
    assert len(provider.requests) == 2
    tool_result = provider.requests[1][-1]
    assert tool_result.role is MessageRole.TOOL
    assert tool_result.tool_name == "get_current_time"
    assert '"ok":true' in tool_result.content


class StreamingToolProvider:
    """Streams a tool request, then a final answer in a second model pass."""

    def __init__(self) -> None:
        self.requests: list[list[ChatMessage]] = []

    async def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        model: str,
        tools: Sequence[ToolDefinition] = (),
    ) -> ChatCompletion:
        raise AssertionError("complete is not used by this test")

    async def stream(
        self,
        messages: Sequence[ChatMessage],
        *,
        model: str,
        tools: Sequence[ToolDefinition] = (),
    ) -> AsyncIterator[ChatStreamEvent]:
        assert tools[0].function["name"] == "get_current_time"
        self.requests.append(list(messages))
        if len(self.requests) == 1:
            yield ChatStreamEvent(
                type="tool_call",
                tool_calls=[ToolCall(function=ToolFunction(name="get_current_time", arguments={}))],
            )
            yield ChatStreamEvent(type="done", model=model)
            return
        yield ChatStreamEvent(type="token", content="The tool result is available.")
        yield ChatStreamEvent(type="done", model=model)

    async def aclose(self) -> None:
        return None


def test_streaming_chat_continues_after_a_tool_call() -> None:
    provider = StreamingToolProvider()
    service = ChatService(Settings(environment="test"), provider)

    async def exercise() -> list[ChatStreamEvent]:
        return [
            event
            async for event in service.stream(
                [ConversationMessage(role="user", content="What time is it?")],
                Brain.FAST,
            )
        ]

    events = asyncio.run(exercise())

    assert [event.type for event in events] == ["tool_call", "tool_result", "token", "done"]
    assert events[1].tool_results[0]["tool_name"] == "get_current_time"
    assert events[1].tool_results[0]["ok"] is True
    assert events[-1].model == "qwen3.6:35b"
    assert len(provider.requests) == 2
    assert provider.requests[1][-1].role is MessageRole.TOOL


def test_chat_selects_the_deep_model() -> None:
    provider = FakeProvider()
    app = create_app(Settings(environment="test"), provider=provider)

    with TestClient(app) as client:
        response = client.post(
            "/chat",
            json={"brain": "deep", "messages": [{"role": "user", "content": "Diagnose this."}]},
        )

    assert response.status_code == 200
    assert response.json()["brain"] == "deep"
    assert provider.model == "qwen3.5:122b-a10b"


def test_chat_registers_weather_with_the_waalwijk_default() -> None:
    provider = FakeProvider()
    app = create_app(Settings(environment="test"), provider=provider)

    with TestClient(app) as client:
        response = client.post("/chat", json={"messages": [{"role": "user", "content": "Hello"}]})

    assert response.status_code == 200
    assert [tool.function["name"] for tool in provider.tools] == [
        "get_current_time",
        "get_current_weather",
    ]


def test_streaming_chat_persists_and_restores_a_conversation_with_tool_activity(tmp_path) -> None:
    provider = StreamingToolProvider()
    app = create_app(
        Settings(environment="test", database_path=str(tmp_path / "atlas.db")), provider=provider
    )

    with TestClient(app) as client:
        created = client.post("/conversations")
        assert created.status_code == 201
        conversation_id = created.json()["id"]

        response = client.post(
            "/chat/stream",
            json={
                "conversation_id": conversation_id,
                "messages": [{"role": "user", "content": "What time is it?"}],
            },
        )
        restored = client.get(f"/conversations/{conversation_id}")
        summaries = client.get("/conversations")

    assert response.status_code == 200
    assert 'event: tool_result' in response.text
    assert restored.status_code == 200
    messages = restored.json()["messages"]
    assert [(message["role"], message["content"]) for message in messages] == [
        ("user", "What time is it?"),
        ("assistant", "The tool result is available."),
    ]
    assert messages[1]["tool_activity"][0]["tool_name"] == "get_current_time"
    assert messages[1]["tool_activity"][0]["ok"] is True
    assert summaries.json()[0]["title"] == "What time is it?"
