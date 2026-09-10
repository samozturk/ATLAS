import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Sequence

from fastapi.testclient import TestClient

from atlas.config import Settings
from atlas.conversations import ConversationStore
from atlas.llm.models import (
    Brain,
    ChatCompletion,
    ChatMessage,
    ChatStreamEvent,
    ConversationMessage,
    MessageRole,
    ToolDefinition,
)
from atlas.llm.service import ChatService
from atlas.main import create_app
from atlas.memory import PersonalMemoryWorker, _parse_candidates


class MemoryProvider:
    """A focused fake that returns the constrained memory-extraction format."""

    def __init__(self) -> None:
        self.requests: list[list[ChatMessage]] = []
        self.tools: list[list[ToolDefinition]] = []

    async def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        model: str,
        tools: Sequence[ToolDefinition] = (),
    ) -> ChatCompletion:
        assert model == "qwen3.6:35b"
        self.requests.append(list(messages))
        self.tools.append(list(tools))
        return ChatCompletion(
            message=ChatMessage(
                role=MessageRole.ASSISTANT,
                content='[{"fact":"The user prefers coffee in the morning.","category":"preference"}]',
            ),
            model=model,
        )

    async def stream(self, *args: object, **kwargs: object):
        yield ChatStreamEvent(type="done")

    async def aclose(self) -> None:
        return None


def test_idle_conversation_adds_a_deduplicated_personal_memory(tmp_path: Path) -> None:
    store = ConversationStore(str(tmp_path / "atlas.db"))
    store.initialize()
    conversation = store.create_conversation()
    store.append_message(conversation.id, role="user", content="I prefer coffee in the morning.")
    store.append_message(conversation.id, role="assistant", content="Noted.")
    provider = MemoryProvider()
    worker = PersonalMemoryWorker(Settings(database_path=str(tmp_path / "atlas.db")), provider, store)

    completed = asyncio.run(
        worker.scan_once(idle_before=datetime.now(timezone.utc) + timedelta(seconds=1))
    )

    assert completed == 1
    assert [memory.fact for memory in store.list_personal_memories()] == [
        "The user prefers coffee in the morning."
    ]
    assert "Noted." not in provider.requests[0][1].content
    assert provider.tools[0] == []
    assert asyncio.run(
        worker.scan_once(idle_before=datetime.now(timezone.utc) + timedelta(seconds=1))
    ) == 0
    assert len(provider.requests) == 1


def test_memory_api_returns_the_local_profile(tmp_path: Path) -> None:
    settings = Settings(database_path=str(tmp_path / "atlas.db"))
    app = create_app(settings, provider=MemoryProvider())

    with TestClient(app) as client:
        conversation = app.state.conversation_store.create_conversation()
        app.state.conversation_store.append_message(
            conversation.id, role="user", content="I use ATLAS at home."
        )
        updated_at = app.state.conversation_store.get_conversation(conversation.id).updated_at
        assert app.state.conversation_store.complete_personal_memory_scan(
            conversation_id=conversation.id,
            expected_updated_at=updated_at,
            memories=[("The user uses ATLAS at home.", "household")],
        )
        response = client.get("/memories")

    assert response.status_code == 200
    assert response.json()[0]["fact"] == "The user uses ATLAS at home."


def test_memory_parser_rejects_sensitive_and_malformed_candidates() -> None:
    assert _parse_candidates("not JSON", limit=8) == []
    assert _parse_candidates(
        '[{"fact":"The user password is letmein","category":"identity"}]', limit=8
    ) == []


def test_reviewed_personal_memory_is_available_to_chat_as_profile_data(tmp_path: Path) -> None:
    provider = MemoryProvider()
    service = ChatService(Settings(database_path=str(tmp_path / "atlas.db")), provider)

    asyncio.run(
        service.reply(
            [ConversationMessage(role="user", content="What should I make for breakfast?")],
            Brain.FAST,
            personal_memory=["The user prefers coffee in the morning."],
        )
    )

    assert provider.requests[0][1].role is MessageRole.SYSTEM
    assert "profile derived from explicit user statements" in provider.requests[0][1].content
    assert "The user prefers coffee in the morning." in provider.requests[0][1].content
