import asyncio
import json

import httpx

from atlas.config import Settings
from atlas.llm.models import ToolCall, ToolFunction
from atlas.tools import ToolRegistry, WebSearchTool
from atlas.web_search import SearxngClient, WebSearchResult


def test_searxng_client_requests_json_and_returns_safe_compact_results() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "title": "  ATLAS   search result ",
                        "url": "https://example.com/atlas",
                        "content": " A compact   result snippet. ",
                        "engine": "duckduckgo",
                        "publishedDate": "2026-10-06",
                    },
                    {"title": "Unsafe", "url": "file:///private/data", "content": "ignored"},
                ]
            },
        )

    async def exercise() -> list[WebSearchResult]:
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://searxng.test")
        search = SearxngClient(Settings(web_search_max_results=3, web_search_safe_search=2), client)
        results = await search.search("ATLAS local assistant", limit=9)
        await client.aclose()
        return results

    results = asyncio.run(exercise())

    assert requests[0].url.path == "/search"
    assert dict(requests[0].url.params) == {
        "q": "ATLAS local assistant",
        "format": "json",
        "categories": "general",
        "safesearch": "2",
    }
    assert results == [
        WebSearchResult(
            title="ATLAS search result",
            url="https://example.com/atlas",
            snippet="A compact result snippet.",
            engine="duckduckgo",
            published_date="2026-10-06",
        )
    ]


def test_web_search_tool_returns_results_as_controlled_json() -> None:
    class StaticSearch:
        async def search(self, query: str, *, limit: int | None = None) -> list[WebSearchResult]:
            assert query == "current local news"
            assert limit == 2
            return [
                WebSearchResult(
                    title="Local news",
                    url="https://example.com/news",
                    snippet="A result.",
                )
            ]

    registry = ToolRegistry([WebSearchTool(StaticSearch())], execution_timeout_seconds=1)

    async def exercise():
        return await registry.execute(
            ToolCall(function=ToolFunction(name="search_web", arguments={"query": "current local news", "limit": 2}))
        )

    result = asyncio.run(exercise())

    assert result.ok is True
    assert json.loads(result.content) == [
        {"title": "Local news", "url": "https://example.com/news", "snippet": "A result."}
    ]
