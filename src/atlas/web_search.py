"""Bounded, read-only SearXNG client for current web information."""

from typing import Any
from urllib.parse import urlparse

import httpx
from pydantic import BaseModel, Field

from atlas.config import Settings


class WebSearchError(Exception):
    """A SearXNG failure that is safe to surface through the tool layer."""


class WebSearchResult(BaseModel):
    """A compact, model-readable result with its original source URL."""

    title: str = Field(min_length=1, max_length=500)
    url: str = Field(min_length=1, max_length=2_000)
    snippet: str = Field(default="", max_length=2_000)
    engine: str | None = Field(default=None, max_length=100)
    published_date: str | None = Field(default=None, max_length=100)


class SearxngClient:
    """Call a self-hosted SearXNG JSON API without exposing its internals."""

    def __init__(self, settings: Settings, client: httpx.AsyncClient | None = None) -> None:
        self._max_results = settings.web_search_max_results
        self._safe_search = settings.web_search_safe_search
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            base_url=settings.searxng_base_url.rstrip("/"),
            timeout=httpx.Timeout(settings.web_search_request_timeout_seconds),
            follow_redirects=True,
        )

    async def search(self, query: str, *, limit: int | None = None) -> list[WebSearchResult]:
        """Return a small, validated set of general web results for a query."""
        requested_limit = min(limit or self._max_results, self._max_results)
        try:
            response = await self._client.get(
                "/search",
                params={
                    "q": query,
                    "format": "json",
                    "categories": "general",
                    "safesearch": self._safe_search,
                },
            )
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPError, TypeError, ValueError) as error:
            raise WebSearchError("Web search is unavailable; try again shortly.") from error
        if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
            raise WebSearchError("Web search returned an invalid response.")
        results: list[WebSearchResult] = []
        for item in payload["results"]:
            result = _result_from_payload(item)
            if result is not None:
                results.append(result)
            if len(results) >= requested_limit:
                break
        return results

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()


def _result_from_payload(item: Any) -> WebSearchResult | None:
    if not isinstance(item, dict):
        return None
    title = item.get("title")
    url = item.get("url")
    if not isinstance(title, str) or not title.strip() or not isinstance(url, str):
        return None
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return None
    snippet = item.get("content", "")
    engine = item.get("engine")
    published_date = item.get("publishedDate")
    return WebSearchResult(
        title=" ".join(title.split()),
        url=url,
        snippet=" ".join(snippet.split()) if isinstance(snippet, str) else "",
        engine=engine if isinstance(engine, str) else None,
        published_date=published_date if isinstance(published_date, str) else None,
    )
