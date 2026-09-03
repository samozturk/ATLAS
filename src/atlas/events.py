"""Core event contract used by future ATLAS adapters and services."""

from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class AtlasEvent(BaseModel):
    """An immutable-enough transport message with traceable metadata."""

    # Use a broker-neutral envelope now so MQTT and NATS adapters can later
    # translate their native messages without changing application consumers.
    id: UUID = Field(default_factory=uuid4)
    type: str = Field(min_length=1, examples=["system.started"])
    source: str = Field(default="atlas", min_length=1)
    occurred_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    # A correlation ID is distinct from the event ID: one user action may fan
    # out into several events while remaining traceable as a single operation.
    correlation_id: UUID = Field(default_factory=uuid4)
    payload: dict[str, Any] = Field(default_factory=dict)
