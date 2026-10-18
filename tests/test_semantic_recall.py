import asyncio
import sqlite3
import struct
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


def test_semantic_recall_reindexes_when_the_embedding_model_changes(tmp_path: Path) -> None:
    store = ConversationStore(str(tmp_path / "atlas.db"))
    store.initialize()
    conversation = store.create_conversation()
    user = store.append_message(conversation.id, role="user", content="My garden project needs watering.")
    assistant = store.append_message(conversation.id, role="assistant", content="I can help track your garden project.")
    turn = SemanticConversationTurn(
        conversation_id=conversation.id,
        user_message=user,
        assistant_message=assistant,
    )

    store.store_semantic_turn(turn, content="old model", embedding=[1.0, 0.0], embedding_model="old")

    assert len(store.list_unindexed_semantic_turns(embedding_model="new")) == 1
    store.store_semantic_turn(turn, content="new model", embedding=[0.0, 1.0], embedding_model="new")

    assert len(store.list_semantic_documents(embedding_model="old")) == 1
    assert len(store.list_semantic_documents(embedding_model="new")) == 1
    assert store.semantic_recall_counts(embedding_model="new") == (1, 0)


def test_deleting_a_conversation_removes_vectors_but_preserves_personal_memory(tmp_path: Path) -> None:
    store = ConversationStore(str(tmp_path / "atlas.db"))
    store.initialize()
    conversation = store.create_conversation()
    user = store.append_message(conversation.id, role="user", content="I prefer coffee in the morning.")
    assistant = store.append_message(conversation.id, role="assistant", content="I will remember that.")
    turn = SemanticConversationTurn(
        conversation_id=conversation.id,
        user_message=user,
        assistant_message=assistant,
    )
    store.store_semantic_turn(turn, content="coffee", embedding=[1.0, 0.0], embedding_model="test")
    updated_at = store.get_conversation(conversation.id).updated_at
    assert store.complete_personal_memory_scan(
        conversation_id=conversation.id,
        expected_updated_at=updated_at,
        memories=[("Prefers coffee in the morning.", "preference")],
    )

    store.delete_conversation(conversation.id)

    assert store.list_conversations() == []
    assert store.list_semantic_documents(embedding_model="test") == []
    assert store.semantic_recall_counts(embedding_model="test") == (0, 0)
    assert store.list_personal_memories()[0].source_conversation_id is None


def test_existing_database_migrates_for_chat_deletion_and_model_reindexing(tmp_path: Path) -> None:
    database_path = tmp_path / "atlas.db"
    connection = sqlite3.connect(database_path)
    connection.execute("PRAGMA foreign_keys = ON")
    connection.executescript(
        """
        CREATE TABLE conversations (
            id TEXT PRIMARY KEY, title TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
        );
        CREATE TABLE messages (
            id TEXT PRIMARY KEY, conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
            role TEXT NOT NULL, content TEXT NOT NULL, brain TEXT, model TEXT, created_at TEXT NOT NULL
        );
        CREATE TABLE personal_memories (
            id TEXT PRIMARY KEY, fact TEXT NOT NULL, category TEXT NOT NULL, fingerprint TEXT NOT NULL UNIQUE,
            source_conversation_id TEXT NOT NULL REFERENCES conversations(id), created_at TEXT NOT NULL,
            last_confirmed_at TEXT NOT NULL
        );
        CREATE TABLE conversation_vectors (
            message_id TEXT PRIMARY KEY REFERENCES messages(id) ON DELETE CASCADE,
            conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
            content TEXT NOT NULL, embedding BLOB NOT NULL, dimensions INTEGER NOT NULL,
            embedding_model TEXT NOT NULL, created_at TEXT NOT NULL
        );
        """
    )
    timestamp = "2026-01-01T00:00:00+00:00"
    connection.execute("INSERT INTO conversations VALUES (?, ?, ?, ?)", ("chat", "Chat", timestamp, timestamp))
    connection.execute(
        "INSERT INTO messages VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("user", "chat", "user", "A saved question", None, None, timestamp),
    )
    connection.execute(
        "INSERT INTO messages VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("assistant", "chat", "assistant", "A saved answer", None, None, "2026-01-01T00:00:01+00:00"),
    )
    connection.execute(
        "INSERT INTO personal_memories VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("memory", "Prefers coffee.", "preference", "fingerprint", "chat", timestamp, timestamp),
    )
    connection.execute(
        "INSERT INTO conversation_vectors VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("assistant", "chat", "old vector", struct.pack("<2f", 1.0, 0.0), 2, "old", timestamp),
    )
    connection.commit()
    connection.close()

    store = ConversationStore(str(database_path))
    store.initialize()

    assert len(store.list_semantic_documents(embedding_model="old")) == 1
    assert len(store.list_unindexed_semantic_turns(embedding_model="new")) == 1
    store.delete_conversation("chat")
    assert store.list_personal_memories()[0].source_conversation_id is None
