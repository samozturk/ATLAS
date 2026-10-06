import asyncio
from collections.abc import Sequence
from pathlib import Path

from atlas.config import Settings
from atlas.conversations import ConversationStore, SemanticConversationTurn
from atlas.semantic_recall import SemanticRecallService


class FakeEmbeddingClient:
    """Small deterministic embedding client for local semantic-recall tests."""

    def __init__(self) -> None:
        self.requests: list[list[str]] = []
        self.closed = False

    async def embed(self, texts: Sequence[str], *, model: str) -> list[list[float]]:
        assert model == "test-embeddings"
        self.requests.append(list(texts))
        return [
            [
                float(text.lower().count("coffee")),
                float(text.lower().count("garden")),
                float(text.lower().count("project")),
                1.0,
            ]
            for text in texts
        ]

    async def aclose(self) -> None:
        self.closed = True


def test_semantic_recall_indexes_and_returns_relevant_prior_exchange(tmp_path: Path) -> None:
    store = ConversationStore(str(tmp_path / "atlas.db"))
    store.initialize()
    first = store.create_conversation()
    user = store.append_message(first.id, role="user", content="I prefer coffee in the morning.")
    assistant = store.append_message(first.id, role="assistant", content="I will remember your coffee preference.")
    embeddings = FakeEmbeddingClient()
    service = SemanticRecallService(
        Settings(
            database_path=str(tmp_path / "atlas.db"),
            embedding_model="test-embeddings",
            semantic_retrieval_min_similarity=0.1,
        ),
        store,
        embeddings,  # type: ignore[arg-type]
    )

    async def exercise() -> list[str]:
        await service.index_turn(
            SemanticConversationTurn(
                conversation_id=first.id,
                user_message=user,
                assistant_message=assistant,
            )
        )
        recalls = await service.recall("What coffee should I have today?")
        await service.aclose()
        return [recall.content for recall in recalls]

    recalls = asyncio.run(exercise())

    assert recalls == ["User: I prefer coffee in the morning.\n\nATLAS: I will remember your coffee preference."]
    assert embeddings.closed is True


def test_semantic_backfill_indexes_existing_completed_turns(tmp_path: Path) -> None:
    store = ConversationStore(str(tmp_path / "atlas.db"))
    store.initialize()
    conversation = store.create_conversation()
    store.append_message(conversation.id, role="user", content="My garden project needs watering.")
    store.append_message(conversation.id, role="assistant", content="I can help track your garden project.")
    embeddings = FakeEmbeddingClient()
    service = SemanticRecallService(
        Settings(database_path=str(tmp_path / "atlas.db"), embedding_model="test-embeddings"),
        store,
        embeddings,  # type: ignore[arg-type]
    )

    assert asyncio.run(service.backfill()) == 1
    assert len(store.list_semantic_documents(embedding_model="test-embeddings")) == 1
    assert asyncio.run(service.backfill()) == 0
