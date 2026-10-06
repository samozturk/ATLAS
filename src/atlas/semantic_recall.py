"""Local vector-backed retrieval of relevant excerpts from earlier conversations."""

import math
from collections.abc import Sequence

import structlog
from pydantic import BaseModel, Field

from atlas.config import Settings
from atlas.conversations import ConversationStore, SemanticConversationTurn
from atlas.embeddings import EmbeddingError, OllamaEmbeddingClient

log = structlog.get_logger(__name__)


class SemanticRecall(BaseModel):
    """A similarity-ranked past conversation excerpt safe to add as context."""

    content: str = Field(min_length=1)
    score: float
    conversation_id: str


class SemanticRecallService:
    """Persist local embeddings and retrieve only small, relevant prior excerpts."""

    def __init__(
        self,
        settings: Settings,
        store: ConversationStore,
        embedding_client: OllamaEmbeddingClient,
    ) -> None:
        self._settings = settings
        self._store = store
        self._embedding_client = embedding_client

    async def index_turn(self, turn: SemanticConversationTurn) -> None:
        """Embed a newly completed exchange without ever failing the chat response."""
        if not self._settings.semantic_retrieval_enabled:
            return
        await self._index_turns([turn])

    async def backfill(self) -> int:
        """Embed historical completed turns in batches when ATLAS starts."""
        if not self._settings.semantic_retrieval_enabled:
            return 0
        indexed = 0
        while turns := self._store.list_unindexed_semantic_turns(
            limit=self._settings.semantic_backfill_batch_size
        ):
            try:
                await self._index_turns(turns)
            except EmbeddingError as error:
                log.warning("semantic_recall.backfill_paused", detail=str(error))
                return indexed
            indexed += len(turns)
        if indexed:
            log.info("semantic_recall.backfill_complete", indexed=indexed)
        return indexed

    async def recall(
        self, query: str, *, exclude_conversation_id: str | None = None
    ) -> list[SemanticRecall]:
        """Find the few historical exchanges that are semantically close to a new question."""
        if not self._settings.semantic_retrieval_enabled or len(query.strip()) < 12:
            return []
        candidates = self._store.list_semantic_documents(
            embedding_model=self._settings.embedding_model,
            exclude_conversation_id=exclude_conversation_id,
        )
        if not candidates:
            return []
        query_vector = (await self._embedding_client.embed([query], model=self._settings.embedding_model))[0]
        ranked = [
            SemanticRecall(
                content=document.content,
                score=_cosine_similarity(query_vector, document.embedding),
                conversation_id=document.conversation_id,
            )
            for document in candidates
            if len(document.embedding) == len(query_vector)
        ]
        return [
            result
            for result in sorted(ranked, key=lambda item: item.score, reverse=True)
            if result.score >= self._settings.semantic_retrieval_min_similarity
        ][: self._settings.semantic_retrieval_limit]

    async def aclose(self) -> None:
        await self._embedding_client.aclose()

    async def _index_turns(self, turns: Sequence[SemanticConversationTurn]) -> None:
        documents = [_turn_content(turn) for turn in turns]
        embeddings = await self._embedding_client.embed(documents, model=self._settings.embedding_model)
        for turn, content, embedding in zip(turns, documents, embeddings, strict=True):
            self._store.store_semantic_turn(
                turn,
                content=content,
                embedding=embedding,
                embedding_model=self._settings.embedding_model,
            )


def _turn_content(turn: SemanticConversationTurn) -> str:
    """Keep an exchange legible in retrieved context and bounded for the embedding model."""
    content = f"User: {turn.user_message.content}\n\nATLAS: {turn.assistant_message.content}"
    return content[:8_000]


def _cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """Use cosine similarity even if an embedding provider is not unit-normalized."""
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    return dot / (left_norm * right_norm) if left_norm and right_norm else 0.0
