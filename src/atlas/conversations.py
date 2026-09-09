"""Local SQLite persistence for ATLAS conversations and audited tool activity."""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field

from atlas.llm.models import Brain, ConversationMessage
from atlas.events import AtlasEvent


class ConversationNotFoundError(LookupError):
    """Raised when a requested local conversation does not exist."""


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
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                    content TEXT NOT NULL,
                    brain TEXT,
                    model TEXT,
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
                """
            )

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

    def get_conversation(self, conversation_id: str) -> StoredConversation:
        with self._connect() as connection:
            conversation = connection.execute(
                "SELECT id, title, created_at, updated_at FROM conversations WHERE id = ?",
                (conversation_id,),
            ).fetchone()
            if conversation is None:
                raise ConversationNotFoundError(conversation_id)
            message_rows = connection.execute(
                "SELECT id, role, content, brain, model, created_at FROM messages "
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
        )
        with self._connect() as connection:
            existing = connection.execute(
                "SELECT title FROM conversations WHERE id = ?", (conversation_id,)
            ).fetchone()
            if existing is None:
                raise ConversationNotFoundError(conversation_id)
            connection.execute(
                "INSERT INTO messages (id, conversation_id, role, content, brain, model, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    message.id,
                    conversation_id,
                    message.role,
                    message.content,
                    message.brain.value if message.brain is not None else None,
                    message.model,
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
