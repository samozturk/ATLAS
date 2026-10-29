"""Local SQLite persistence for ATLAS conversations and audited tool activity."""

import json
import sqlite3
import struct
from hashlib import sha256
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field

from atlas.llm.models import Brain, ConversationMessage, RecallSource
from atlas.events import AtlasEvent


class ConversationNotFoundError(LookupError):
    """Raised when a requested local conversation does not exist."""


class PersonalMemoryNotFoundError(LookupError):
    """Raised when a requested local profile fact does not exist."""


class DuplicatePersonalMemoryError(ValueError):
    """Raised when an edit would duplicate another saved profile fact."""


class StoredToolActivity(BaseModel):
    """One tool call made while producing a stored assistant response."""

    tool_call_id: str | None = None
    tool_name: str = Field(min_length=1)
    arguments: dict[str, Any] = Field(default_factory=dict)
    ok: bool | None = None
    content: str | None = None


class StoredMessage(BaseModel):
    """A user or assistant message in a persisted conversation."""

    id: str
    role: Literal["user", "assistant"]
    content: str
    brain: Brain | None = None
    model: str | None = None
    created_at: datetime
    tool_activity: list[StoredToolActivity] = Field(default_factory=list)
    # This is deliberately provenance only. Recalled excerpts remain transient
    # context for the model and are never stored alongside the response.
    recall_sources: list[RecallSource] = Field(default_factory=list)

    def as_conversation_message(self) -> ConversationMessage:
        return ConversationMessage(role=self.role, content=self.content)


class ConversationSummary(BaseModel):
    """The lightweight record used by the conversation sidebar."""

    id: str
    title: str
    created_at: datetime
    updated_at: datetime


class StoredConversation(ConversationSummary):
    """A conversation and all messages needed to resume it."""

    messages: list[StoredMessage]


class StoredEvent(AtlasEvent):
    """An incoming home event retained for the local event feed and audit history."""

    received_at: datetime


class PersonalMemory(BaseModel):
    """A reviewed, local fact ATLAS may use to maintain continuity."""

    id: str
    fact: str
    category: str
    # A user may remove the source chat while retaining their separately
    # managed profile, in which case this provenance is intentionally blank.
    source_conversation_id: str | None = None
    created_at: datetime
    last_confirmed_at: datetime


class PendingPersonalMemoryScan(BaseModel):
    """An idle conversation whose user messages have not yet been reviewed."""

    conversation_id: str
    updated_at: datetime
    user_messages: list[StoredMessage]


class SemanticConversationTurn(BaseModel):
    """One completed exchange ready to be represented as a local vector."""

    conversation_id: str
    user_message: StoredMessage
    assistant_message: StoredMessage


class StoredSemanticDocument(BaseModel):
    """A persisted conversation exchange and its local embedding vector."""

    message_id: str
    conversation_id: str
    content: str
    embedding: list[float]
    embedding_model: str
    created_at: datetime


class ConversationStore:
    """Own the SQLite schema and keep all persistent state on the ATLAS host."""

    def __init__(self, database_path: str) -> None:
        self._database_path = Path(database_path)

    def initialize(self) -> None:
        """Create the local database and schema before the service is ready."""
        self._database_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    memory_scanned_at TEXT
                );
                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                    content TEXT NOT NULL,
                    brain TEXT,
                    model TEXT,
                    recall_sources_json TEXT NOT NULL DEFAULT '[]',
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS messages_conversation_created_at
                    ON messages(conversation_id, created_at);
                CREATE TABLE IF NOT EXISTS tool_activity (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    message_id TEXT NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
                    tool_call_id TEXT,
                    tool_name TEXT NOT NULL,
                    arguments_json TEXT NOT NULL,
                    ok INTEGER,
                    content TEXT
                );
                CREATE INDEX IF NOT EXISTS tool_activity_message_id
                    ON tool_activity(message_id, id);
                CREATE TABLE IF NOT EXISTS events (
                    id TEXT PRIMARY KEY,
                    type TEXT NOT NULL,
                    source TEXT NOT NULL,
                    occurred_at TEXT NOT NULL,
                    correlation_id TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    received_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS events_received_at ON events(received_at DESC);
                CREATE TABLE IF NOT EXISTS personal_memories (
                    id TEXT PRIMARY KEY,
                    fact TEXT NOT NULL,
                    category TEXT NOT NULL,
                    fingerprint TEXT NOT NULL UNIQUE,
                    source_conversation_id TEXT,
                    created_at TEXT NOT NULL,
                    last_confirmed_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS personal_memories_last_confirmed_at
                    ON personal_memories(last_confirmed_at DESC);
                CREATE TABLE IF NOT EXISTS conversation_vectors (
                    message_id TEXT NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
                    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                    content TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    dimensions INTEGER NOT NULL,
                    embedding_model TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (message_id, embedding_model)
                );
                CREATE INDEX IF NOT EXISTS conversation_vectors_model_created_at
                    ON conversation_vectors(embedding_model, created_at DESC);
                """
            )
            # Existing installations predate memory_scanned_at. SQLite does not
            # support ADD COLUMN IF NOT EXISTS, so make this migration explicit.
            columns = {row["name"] for row in connection.execute("PRAGMA table_info(conversations)")}
            if "memory_scanned_at" not in columns:
                connection.execute("ALTER TABLE conversations ADD COLUMN memory_scanned_at TEXT")
            message_columns = {row["name"] for row in connection.execute("PRAGMA table_info(messages)")}
            if "recall_sources_json" not in message_columns:
                connection.execute(
                    "ALTER TABLE messages ADD COLUMN recall_sources_json TEXT NOT NULL DEFAULT '[]'"
                )
            self._migrate_personal_memories(connection)
            self._migrate_conversation_vectors(connection)
            connection.execute("PRAGMA optimize")

    def create_conversation(self) -> ConversationSummary:
        now = _now()
        summary = ConversationSummary(
            id=str(uuid4()),
            title="New conversation",
            created_at=now,
            updated_at=now,
        )
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO conversations (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (summary.id, summary.title, _dump_time(now), _dump_time(now)),
            )
        return summary

    def list_conversations(self) -> list[ConversationSummary]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT id, title, created_at, updated_at FROM conversations "
                "ORDER BY updated_at DESC, created_at DESC"
            ).fetchall()
        return [self._summary_from_row(row) for row in rows]

    def delete_conversation(self, conversation_id: str) -> None:
        """Delete one chat and its dependent history without deleting profile facts."""
        with self._connect() as connection:
            exists = connection.execute(
                "SELECT 1 FROM conversations WHERE id = ?", (conversation_id,)
            ).fetchone()
            if exists is None:
                raise ConversationNotFoundError(conversation_id)
            # Memories are a separate user-visible profile. Preserve them while
            # removing a deleted chat as their now-unavailable provenance.
            connection.execute(
                "UPDATE personal_memories SET source_conversation_id = NULL "
                "WHERE source_conversation_id = ?",
                (conversation_id,),
            )
            connection.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))

    def get_conversation(self, conversation_id: str) -> StoredConversation:
        with self._connect() as connection:
            conversation = connection.execute(
                "SELECT id, title, created_at, updated_at FROM conversations WHERE id = ?",
                (conversation_id,),
            ).fetchone()
            if conversation is None:
                raise ConversationNotFoundError(conversation_id)
            message_rows = connection.execute(
                "SELECT id, role, content, brain, model, recall_sources_json, created_at FROM messages "
                "WHERE conversation_id = ? ORDER BY created_at, rowid",
                (conversation_id,),
            ).fetchall()
            messages = [self._message_from_row(connection, row) for row in message_rows]
        return StoredConversation(**self._summary_from_row(conversation).model_dump(), messages=messages)

    def append_message(
        self,
        conversation_id: str,
        *,
        role: Literal["user", "assistant"],
        content: str,
        brain: Brain | None = None,
        model: str | None = None,
        tool_activity: list[StoredToolActivity] | None = None,
        recall_sources: list[RecallSource] | None = None,
    ) -> StoredMessage:
        now = _now()
        message = StoredMessage(
            id=str(uuid4()),
            role=role,
            content=content,
            brain=brain,
            model=model,
            created_at=now,
            tool_activity=tool_activity or [],
            recall_sources=recall_sources or [],
        )
        with self._connect() as connection:
            existing = connection.execute(
                "SELECT title FROM conversations WHERE id = ?", (conversation_id,)
            ).fetchone()
            if existing is None:
                raise ConversationNotFoundError(conversation_id)
            connection.execute(
                "INSERT INTO messages "
                "(id, conversation_id, role, content, brain, model, recall_sources_json, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    message.id,
                    conversation_id,
                    message.role,
                    message.content,
                    message.brain.value if message.brain is not None else None,
                    message.model,
                    json.dumps(
                        [source.model_dump(mode="json") for source in message.recall_sources],
                        separators=(",", ":"),
                    ),
                    _dump_time(message.created_at),
                ),
            )
            for tool in message.tool_activity:
                connection.execute(
                    "INSERT INTO tool_activity "
                    "(message_id, tool_call_id, tool_name, arguments_json, ok, content) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        message.id,
                        tool.tool_call_id,
                        tool.tool_name,
                        json.dumps(tool.arguments, separators=(",", ":")),
                        None if tool.ok is None else int(tool.ok),
                        tool.content,
                    ),
                )
            title = _title_for(content) if role == "user" and existing["title"] == "New conversation" else existing["title"]
            connection.execute(
                "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?",
                (title, _dump_time(now), conversation_id),
            )
        return message

    def record_event(self, event: AtlasEvent) -> StoredEvent:
        """Persist one normalized event before any future rule or LLM consumer sees it."""
        received_at = _now()
        stored_event = StoredEvent(**event.model_dump(), received_at=received_at)
        with self._connect() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO events "
                "(id, type, source, occurred_at, correlation_id, payload_json, received_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    str(stored_event.id),
                    stored_event.type,
                    stored_event.source,
                    _dump_time(stored_event.occurred_at),
                    str(stored_event.correlation_id),
                    json.dumps(stored_event.payload, separators=(",", ":")),
                    _dump_time(stored_event.received_at),
                ),
            )
        return stored_event

    def list_events(self, *, limit: int = 50) -> list[StoredEvent]:
        """Read the newest local home events without exposing broker internals."""
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT id, type, source, occurred_at, correlation_id, payload_json, received_at "
                "FROM events ORDER BY received_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [
            StoredEvent(
                id=row["id"],
                type=row["type"],
                source=row["source"],
                occurred_at=_load_time(row["occurred_at"]),
                correlation_id=row["correlation_id"],
                payload=json.loads(row["payload_json"]),
                received_at=_load_time(row["received_at"]),
            )
            for row in rows
        ]

    def pending_personal_memory_scans(
        self, *, idle_before: datetime, limit: int = 25
    ) -> list[PendingPersonalMemoryScan]:
        """Return idle conversations with user turns not yet reviewed for memory."""
        with self._connect() as connection:
            conversations = connection.execute(
                "SELECT id, updated_at FROM conversations "
                "WHERE updated_at <= ? AND (memory_scanned_at IS NULL OR memory_scanned_at < updated_at) "
                "ORDER BY updated_at ASC LIMIT ?",
                (_dump_time(idle_before), limit),
            ).fetchall()
            scans: list[PendingPersonalMemoryScan] = []
            for conversation in conversations:
                messages = connection.execute(
                    "SELECT id, role, content, brain, model, created_at FROM messages "
                    "WHERE conversation_id = ? AND role = 'user' ORDER BY created_at, rowid",
                    (conversation["id"],),
                ).fetchall()
                scans.append(
                    PendingPersonalMemoryScan(
                        conversation_id=conversation["id"],
                        updated_at=_load_time(conversation["updated_at"]),
                        user_messages=[self._message_from_row(connection, message) for message in messages],
                    )
                )
        return scans

    def complete_personal_memory_scan(
        self,
        *,
        conversation_id: str,
        expected_updated_at: datetime,
        memories: list[tuple[str, str]],
    ) -> bool:
        """Persist reviewed facts if the conversation stayed idle while it was scanned.

        The optimistic updated_at check means a new user message always wins over
        a background result produced from an older snapshot.
        """
        now = _now()
        with self._connect() as connection:
            current = connection.execute(
                "SELECT updated_at FROM conversations WHERE id = ?", (conversation_id,)
            ).fetchone()
            if current is None or current["updated_at"] != _dump_time(expected_updated_at):
                return False
            for fact, category in memories:
                normalized = " ".join(fact.casefold().split())
                if not normalized:
                    continue
                fingerprint = sha256(normalized.encode()).hexdigest()
                existing = connection.execute(
                    "SELECT id FROM personal_memories WHERE fingerprint = ?", (fingerprint,)
                ).fetchone()
                if existing is None:
                    connection.execute(
                        "INSERT INTO personal_memories "
                        "(id, fact, category, fingerprint, source_conversation_id, created_at, last_confirmed_at) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (
                            str(uuid4()),
                            fact,
                            category,
                            fingerprint,
                            conversation_id,
                            _dump_time(now),
                            _dump_time(now),
                        ),
                    )
                else:
                    connection.execute(
                        "UPDATE personal_memories SET last_confirmed_at = ? WHERE id = ?",
                        (_dump_time(now), existing["id"]),
                    )
            connection.execute(
                "UPDATE conversations SET memory_scanned_at = ? WHERE id = ?",
                (_dump_time(now), conversation_id),
            )
        return True

    def list_personal_memories(self, *, limit: int = 100) -> list[PersonalMemory]:
        """Read the user-visible personal-memory record from local SQLite."""
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT id, fact, category, source_conversation_id, created_at, last_confirmed_at "
                "FROM personal_memories ORDER BY last_confirmed_at DESC, created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [
            PersonalMemory(
                id=row["id"],
                fact=row["fact"],
                category=row["category"],
                source_conversation_id=row["source_conversation_id"],
                created_at=_load_time(row["created_at"]),
                last_confirmed_at=_load_time(row["last_confirmed_at"]),
            )
            for row in rows
        ]

    def update_personal_memory(self, memory_id: str, *, fact: str, category: str) -> PersonalMemory:
        """Edit one reviewed profile fact without retaining a duplicate fingerprint."""
        normalized_fact = " ".join(fact.split())
        normalized_category = " ".join(category.split())
        if not normalized_fact or not normalized_category:
            raise ValueError("memory fact and category cannot be blank")
        fingerprint = sha256(normalized_fact.casefold().encode()).hexdigest()
        now = _now()
        with self._connect() as connection:
            existing = connection.execute(
                "SELECT id FROM personal_memories WHERE id = ?", (memory_id,)
            ).fetchone()
            if existing is None:
                raise PersonalMemoryNotFoundError(memory_id)
            duplicate = connection.execute(
                "SELECT id FROM personal_memories WHERE fingerprint = ? AND id != ?",
                (fingerprint, memory_id),
            ).fetchone()
            if duplicate is not None:
                raise DuplicatePersonalMemoryError("An identical memory already exists.")
            connection.execute(
                "UPDATE personal_memories SET fact = ?, category = ?, fingerprint = ?, last_confirmed_at = ? "
                "WHERE id = ?",
                (normalized_fact, normalized_category, fingerprint, _dump_time(now), memory_id),
            )
            row = connection.execute(
                "SELECT id, fact, category, source_conversation_id, created_at, last_confirmed_at "
                "FROM personal_memories WHERE id = ?",
                (memory_id,),
            ).fetchone()
        assert row is not None
        return PersonalMemory(
            id=row["id"],
            fact=row["fact"],
            category=row["category"],
            source_conversation_id=row["source_conversation_id"],
            created_at=_load_time(row["created_at"]),
            last_confirmed_at=_load_time(row["last_confirmed_at"]),
        )

    def delete_personal_memory(self, memory_id: str) -> None:
        """Remove a profile fact that the user no longer wants ATLAS to use."""
        with self._connect() as connection:
            result = connection.execute("DELETE FROM personal_memories WHERE id = ?", (memory_id,))
            if result.rowcount == 0:
                raise PersonalMemoryNotFoundError(memory_id)

    def store_semantic_turn(
        self,
        turn: SemanticConversationTurn,
        *,
        content: str,
        embedding: list[float],
        embedding_model: str,
    ) -> None:
        """Upsert an assistant-completed exchange using the embedding model that made it."""
        if not embedding:
            raise ValueError("semantic embeddings cannot be empty")
        vector = sqlite3.Binary(struct.pack(f"<{len(embedding)}f", *embedding))
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO conversation_vectors "
                "(message_id, conversation_id, content, embedding, dimensions, embedding_model, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(message_id, embedding_model) DO UPDATE SET content = excluded.content, embedding = excluded.embedding, "
                "dimensions = excluded.dimensions, embedding_model = excluded.embedding_model, "
                "created_at = excluded.created_at",
                (
                    turn.assistant_message.id,
                    turn.conversation_id,
                    content,
                    vector,
                    len(embedding),
                    embedding_model,
                    _dump_time(turn.assistant_message.created_at),
                ),
            )

    def list_semantic_documents(
        self,
        *,
        embedding_model: str,
        exclude_conversation_id: str | None = None,
        limit: int = 5_000,
    ) -> list[StoredSemanticDocument]:
        """Read a bounded candidate set for in-process cosine similarity ranking."""
        query = (
            "SELECT message_id, conversation_id, content, embedding, dimensions, embedding_model, created_at "
            "FROM conversation_vectors WHERE embedding_model = ?"
        )
        params: list[object] = [embedding_model]
        if exclude_conversation_id is not None:
            query += " AND conversation_id != ?"
            params.append(exclude_conversation_id)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        with self._connect() as connection:
            rows = connection.execute(query, params).fetchall()
        documents: list[StoredSemanticDocument] = []
        for row in rows:
            dimensions = row["dimensions"]
            encoded = row["embedding"]
            if not isinstance(dimensions, int) or not isinstance(encoded, bytes) or len(encoded) != dimensions * 4:
                continue
            documents.append(
                StoredSemanticDocument(
                    message_id=row["message_id"],
                    conversation_id=row["conversation_id"],
                    content=row["content"],
                    embedding=list(struct.unpack(f"<{dimensions}f", encoded)),
                    embedding_model=row["embedding_model"],
                    created_at=_load_time(row["created_at"]),
                )
            )
        return documents

    def list_unindexed_semantic_turns(
        self, *, embedding_model: str, limit: int = 12
    ) -> list[SemanticConversationTurn]:
        """Find historical completed turns that have not been embedded yet."""
        with self._connect() as connection:
            assistant_rows = connection.execute(
                "SELECT m.id, m.conversation_id, m.role, m.content, m.brain, m.model, m.created_at, m.rowid "
                "FROM messages m LEFT JOIN conversation_vectors v "
                "ON v.message_id = m.id AND v.embedding_model = ? "
                "WHERE m.role = 'assistant' AND v.message_id IS NULL AND m.content != '' "
                "ORDER BY m.created_at, m.rowid LIMIT ?",
                (embedding_model, limit),
            ).fetchall()
            turns: list[SemanticConversationTurn] = []
            for assistant in assistant_rows:
                user = connection.execute(
                    "SELECT id, role, content, brain, model, created_at FROM messages "
                    "WHERE conversation_id = ? AND role = 'user' "
                    "AND (created_at < ? OR (created_at = ? AND rowid < ?)) "
                    "ORDER BY created_at DESC, rowid DESC LIMIT 1",
                    (
                        assistant["conversation_id"],
                        assistant["created_at"],
                        assistant["created_at"],
                        assistant["rowid"],
                    ),
                ).fetchone()
                if user is None:
                    continue
                turns.append(
                    SemanticConversationTurn(
                        conversation_id=assistant["conversation_id"],
                        user_message=self._message_from_row(connection, user),
                        assistant_message=self._message_from_row(connection, assistant),
                    )
                )
        return turns

    def semantic_recall_counts(self, *, embedding_model: str) -> tuple[int, int]:
        """Return completed exchanges indexed and pending for one embedding model."""
        with self._connect() as connection:
            indexed = connection.execute(
                "SELECT COUNT(*) FROM conversation_vectors WHERE embedding_model = ?",
                (embedding_model,),
            ).fetchone()[0]
            pending = connection.execute(
                "SELECT COUNT(*) FROM messages m LEFT JOIN conversation_vectors v "
                "ON v.message_id = m.id AND v.embedding_model = ? "
                "WHERE m.role = 'assistant' AND m.content != '' AND v.message_id IS NULL",
                (embedding_model,),
            ).fetchone()[0]
        return int(indexed), int(pending)

    @staticmethod
    def _migrate_personal_memories(connection: sqlite3.Connection) -> None:
        """Make legacy personal-memory provenance optional before chat deletion."""
        columns = {
            row["name"]: row for row in connection.execute("PRAGMA table_info(personal_memories)")
        }
        source = columns.get("source_conversation_id")
        has_source_foreign_key = bool(
            connection.execute("PRAGMA foreign_key_list(personal_memories)").fetchall()
        )
        if source is None or (not source["notnull"] and not has_source_foreign_key):
            return
        connection.execute("DROP INDEX IF EXISTS personal_memories_last_confirmed_at")
        connection.execute("ALTER TABLE personal_memories RENAME TO personal_memories_legacy")
        connection.execute(
            "CREATE TABLE personal_memories ("
            "id TEXT PRIMARY KEY, fact TEXT NOT NULL, category TEXT NOT NULL, "
            "fingerprint TEXT NOT NULL UNIQUE, source_conversation_id TEXT, "
            "created_at TEXT NOT NULL, last_confirmed_at TEXT NOT NULL)"
        )
        connection.execute(
            "INSERT INTO personal_memories "
            "(id, fact, category, fingerprint, source_conversation_id, created_at, last_confirmed_at) "
            "SELECT id, fact, category, fingerprint, source_conversation_id, created_at, last_confirmed_at "
            "FROM personal_memories_legacy"
        )
        connection.execute("DROP TABLE personal_memories_legacy")
        connection.execute(
            "CREATE INDEX personal_memories_last_confirmed_at "
            "ON personal_memories(last_confirmed_at DESC)"
        )

    @staticmethod
    def _migrate_conversation_vectors(connection: sqlite3.Connection) -> None:
        """Allow a new embedding model to reindex without losing old vectors."""
        columns = {
            row["name"]: row for row in connection.execute("PRAGMA table_info(conversation_vectors)")
        }
        message = columns.get("message_id")
        embedding_model = columns.get("embedding_model")
        if message is None or embedding_model is None or (message["pk"] == 1 and embedding_model["pk"] == 2):
            return
        connection.execute("DROP INDEX IF EXISTS conversation_vectors_model_created_at")
        connection.execute("ALTER TABLE conversation_vectors RENAME TO conversation_vectors_legacy")
        connection.execute(
            "CREATE TABLE conversation_vectors ("
            "message_id TEXT NOT NULL REFERENCES messages(id) ON DELETE CASCADE, "
            "conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE, "
            "content TEXT NOT NULL, embedding BLOB NOT NULL, dimensions INTEGER NOT NULL, "
            "embedding_model TEXT NOT NULL, created_at TEXT NOT NULL, "
            "PRIMARY KEY (message_id, embedding_model))"
        )
        connection.execute(
            "INSERT INTO conversation_vectors "
            "(message_id, conversation_id, content, embedding, dimensions, embedding_model, created_at) "
            "SELECT message_id, conversation_id, content, embedding, dimensions, embedding_model, created_at "
            "FROM conversation_vectors_legacy"
        )
        connection.execute("DROP TABLE conversation_vectors_legacy")
        connection.execute(
            "CREATE INDEX conversation_vectors_model_created_at "
            "ON conversation_vectors(embedding_model, created_at DESC)"
        )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._database_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    @staticmethod
    def _summary_from_row(row: sqlite3.Row) -> ConversationSummary:
        return ConversationSummary(
            id=row["id"],
            title=row["title"],
            created_at=_load_time(row["created_at"]),
            updated_at=_load_time(row["updated_at"]),
        )

    @staticmethod
    def _message_from_row(connection: sqlite3.Connection, row: sqlite3.Row) -> StoredMessage:
        tool_rows = connection.execute(
            "SELECT tool_call_id, tool_name, arguments_json, ok, content "
            "FROM tool_activity WHERE message_id = ? ORDER BY id",
            (row["id"],),
        ).fetchall()
        raw_recall_sources = row["recall_sources_json"] if "recall_sources_json" in row.keys() else "[]"
        try:
            serialized_sources = json.loads(raw_recall_sources)
        except (TypeError, json.JSONDecodeError):
            serialized_sources = []
        recall_sources: list[RecallSource] = []
        if isinstance(serialized_sources, list):
            for source in serialized_sources:
                try:
                    recall_sources.append(RecallSource.model_validate(source))
                except ValueError:
                    continue
        return StoredMessage(
            id=row["id"],
            role=row["role"],
            content=row["content"],
            brain=Brain(row["brain"]) if row["brain"] is not None else None,
            model=row["model"],
            created_at=_load_time(row["created_at"]),
            tool_activity=[
                StoredToolActivity(
                    tool_call_id=tool_row["tool_call_id"],
                    tool_name=tool_row["tool_name"],
                    arguments=json.loads(tool_row["arguments_json"]),
                    ok=None if tool_row["ok"] is None else bool(tool_row["ok"]),
                    content=tool_row["content"],
                )
                for tool_row in tool_rows
            ],
            recall_sources=recall_sources,
        )


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _dump_time(value: datetime) -> str:
    return value.isoformat()


def _load_time(value: str) -> datetime:
    return datetime.fromisoformat(value)


def _title_for(content: str) -> str:
    compact = " ".join(content.split())
    return f"{compact[:57]}…" if len(compact) > 58 else compact
