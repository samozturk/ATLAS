"""Local Ollama embedding client used for semantic recall."""

import math
from collections.abc import Sequence

import httpx

from atlas.config import Settings


class EmbeddingError(Exception):
    """The local embedding endpoint could not produce usable vectors."""


class OllamaEmbeddingClient:
    """Call Ollama's local /api/embed endpoint without introducing a cloud service."""

    def __init__(self, settings: Settings, client: httpx.AsyncClient | None = None) -> None:
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            base_url=settings.ollama_base_url.rstrip("/"),
            timeout=httpx.Timeout(
                timeout=settings.embedding_request_timeout_seconds,
                connect=settings.llm_connect_timeout_seconds,
            ),
        )

    async def embed(self, texts: Sequence[str], *, model: str) -> list[list[float]]:
        """Embed one or more texts in a single bounded local request."""
        if not texts:
            return []
        try:
            response = await self._client.post(
                "/api/embed", json={"model": model, "input": list(texts), "truncate": True}
            )
        except (httpx.TransportError, httpx.TimeoutException) as error:
            raise EmbeddingError("Ollama embeddings are unavailable") from error
        if response.is_error:
            raise EmbeddingError(f"Ollama embeddings failed with HTTP {response.status_code}")
        try:
            body = response.json()
        except ValueError as error:
            raise EmbeddingError("Ollama embeddings returned invalid JSON") from error
        raw_embeddings = body.get("embeddings") if isinstance(body, dict) else None
        if not isinstance(raw_embeddings, list) or len(raw_embeddings) != len(texts):
            raise EmbeddingError("Ollama embeddings returned an unexpected response")
        embeddings: list[list[float]] = []
        dimensions: int | None = None
        for raw_embedding in raw_embeddings:
            if not isinstance(raw_embedding, list) or not raw_embedding:
                raise EmbeddingError("Ollama embeddings returned an empty vector")
            if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in raw_embedding):
                raise EmbeddingError("Ollama embeddings returned an invalid vector")
            vector = [float(value) for value in raw_embedding]
            if dimensions is None:
                dimensions = len(vector)
            elif len(vector) != dimensions:
                raise EmbeddingError("Ollama embeddings returned inconsistent vector dimensions")
            embeddings.append(vector)
        return embeddings

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()
