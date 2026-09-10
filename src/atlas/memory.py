"""Idle-time extraction of an inspectable, local personal-memory profile."""

import asyncio
import json
import re
from datetime import datetime, timedelta, timezone
from typing import Literal

import structlog
from pydantic import BaseModel, Field, ValidationError

from atlas.config import Settings
from atlas.conversations import ConversationStore, PendingPersonalMemoryScan
from atlas.llm.models import ChatMessage, MessageRole
from atlas.llm.provider import LLMError, LLMProvider

log = structlog.get_logger(__name__)


class MemoryCandidate(BaseModel):
    """One constrained candidate returned by the local model."""

    fact: str = Field(min_length=3, max_length=400)
    category: Literal["identity", "preference", "household", "project", "other"]


class PersonalMemoryWorker:
    """Turn explicit, durable user facts into a small local profile after idle time."""

    def __init__(self, settings: Settings, provider: LLMProvider, store: ConversationStore) -> None:
        self._settings = settings
        self._provider = provider
        self._store = store

    async def run(self) -> None:
        """Run until application shutdown; failures leave the conversation eligible to retry."""
        while True:
            try:
                await self.scan_once()
            except Exception:
                log.exception("memory.scan_cycle_failed")
            await asyncio.sleep(self._settings.memory_scan_interval_seconds)

    async def scan_once(self, *, idle_before: datetime | None = None) -> int:
        """Review each eligible idle conversation once and return its completed count."""
        cutoff = idle_before or (
            datetime.now(timezone.utc) - timedelta(seconds=self._settings.memory_idle_seconds)
        )
        completed = 0
        for scan in self._store.pending_personal_memory_scans(idle_before=cutoff):
            try:
                candidates = await self._extract(scan)
            except LLMError:
                log.warning("memory.extraction_failed", conversation_id=scan.conversation_id)
                continue
            except Exception:
                log.exception("memory.extraction_failed", conversation_id=scan.conversation_id)
                continue
            if self._store.complete_personal_memory_scan(
                conversation_id=scan.conversation_id,
                expected_updated_at=scan.updated_at,
                memories=[(candidate.fact, candidate.category) for candidate in candidates],
            ):
                completed += 1
        return completed

    async def _extract(self, scan: PendingPersonalMemoryScan) -> list[MemoryCandidate]:
        if not scan.user_messages:
            return []
        transcript = "\n\n".join(f"User: {message.content}" for message in scan.user_messages)
        # Bound the independent background prompt so one long chat cannot crowd
        # out the normal local assistant or turn into an unbounded request.
        transcript = transcript[-12_000:]
        completion = await self._provider.complete(
            [
                ChatMessage(
                    role=MessageRole.SYSTEM,
                    content=(
                        "Extract only durable facts the user explicitly stated about themselves. "
                        "Do not infer, speculate, summarize temporary requests, or include facts about other people. "
                        "Never retain passwords, tokens, financial details, medical or legal information, "
                        "or precise/live location. Useful examples are voluntarily stated identity, preferences, "
                        "household conventions, long-lived projects, or routines. Return only a JSON array with at "
                        f"most {self._settings.memory_max_facts_per_conversation} objects shaped exactly as "
                        '{"fact":"...","category":"identity|preference|household|project|other"}. '
                        "Use an empty array when there are no safe, explicit facts."
                    ),
                ),
                ChatMessage(role=MessageRole.USER, content=transcript),
            ],
            model=self._settings.fast_model,
            tools=(),
        )
        return _parse_candidates(completion.message.content, self._settings.memory_max_facts_per_conversation)


def _parse_candidates(content: str, limit: int) -> list[MemoryCandidate]:
    """Treat malformed or unsafe model output as no memory rather than storing it."""
    raw = content.strip()
    if raw.startswith("```") and raw.endswith("```"):
        raw = raw.split("\n", 1)[1] if "\n" in raw else ""
        raw = raw.rsplit("```", 1)[0].strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        log.warning("memory.invalid_model_output")
        return []
    if not isinstance(data, list):
        return []
    candidates: list[MemoryCandidate] = []
    seen: set[str] = set()
    for item in data:
        try:
            candidate = MemoryCandidate.model_validate(item)
        except ValidationError:
            continue
        fact = " ".join(candidate.fact.split())
        normalized = fact.casefold()
        if normalized in seen or not _is_safe_fact(fact):
            continue
        seen.add(normalized)
        candidates.append(candidate.model_copy(update={"fact": fact}))
        if len(candidates) >= limit:
            break
    return candidates


def _is_safe_fact(fact: str) -> bool:
    """A second deterministic privacy boundary in case the model misses the prompt."""
    blocked = (
        "password",
        "passcode",
        "access token",
        "api key",
        "credit card",
        "bank account",
        "medical",
        "diagnosis",
        "prescription",
        "lawsuit",
        "legal case",
        "current location",
        "latitude",
        "longitude",
    )
    lowered = fact.casefold()
    if any(term in lowered for term in blocked):
        return False
    # Credential-shaped strings are never a useful personal-memory fact.
    return re.search(r"\b(?:[a-z0-9_-]{20,}|\d[ -]?){12,}\b", fact, flags=re.IGNORECASE) is None
