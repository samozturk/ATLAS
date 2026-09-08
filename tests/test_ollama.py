import asyncio
import json

import httpx

from atlas.config import Settings
from atlas.llm.models import ChatMessage, MessageRole
from atlas.llm.ollama import OllamaProvider
from atlas.llm.provider import LLMUnavailableError


def test_ollama_provider_translates_messages_and_response() -> None:
    request_payload: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        request_payload.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "model": "qwen3:8b",
                "message": {"role": "assistant", "content": "Hi."},
                "prompt_eval_count": 12,
                "eval_count": 3,
            },
        )

    async def exercise() -> None:
        client = httpx.AsyncClient(
            base_url="http://ollama.test",
            transport=httpx.MockTransport(handler),
        )
        provider = OllamaProvider(Settings(), client=client)
        completion = await provider.complete(
            [ChatMessage(role=MessageRole.USER, content="Hello")],
            model="qwen3.6:35b",
        )
        await client.aclose()

        assert completion.message.content == "Hi."
        assert completion.model == "qwen3:8b"
        assert completion.usage.prompt_tokens == 12
        assert completion.usage.completion_tokens == 3

    asyncio.run(exercise())

    assert request_payload == {
        "model": "qwen3.6:35b",
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": False,
    }


def test_ollama_provider_retries_transient_failures() -> None:
    attempts = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(503, json={"error": "model is loading"})

    async def exercise() -> None:
        client = httpx.AsyncClient(
            base_url="http://ollama.test",
            transport=httpx.MockTransport(handler),
        )
        provider = OllamaProvider(
            Settings(llm_max_retries=2, llm_retry_backoff_seconds=0),
            client=client,
        )
        try:
            await provider.complete(
                [ChatMessage(role=MessageRole.USER, content="Hello")],
                model="qwen3.6:35b",
            )
        except LLMUnavailableError:
            pass
        else:
            raise AssertionError("expected a transient Ollama failure")
        finally:
            await client.aclose()

    asyncio.run(exercise())
    assert attempts == 3


def test_ollama_provider_streams_normalized_events() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=(
                b'{"model":"qwen3:8b","message":{"role":"assistant","content":"Hi"}}\n'
                b'{"model":"qwen3:8b","done":true,"prompt_eval_count":4,"eval_count":1}\n'
            ),
        )

    async def exercise() -> list[tuple[str, str, str | None]]:
        client = httpx.AsyncClient(
            base_url="http://ollama.test",
            transport=httpx.MockTransport(handler),
        )
        provider = OllamaProvider(Settings(), client=client)
        events = [
            (event.type, event.content, event.model)
            async for event in provider.stream(
                [ChatMessage(role=MessageRole.USER, content="Hello")],
                model="qwen3.6:35b",
            )
        ]
        await client.aclose()
        return events

    assert asyncio.run(exercise()) == [("token", "Hi", None), ("done", "", "qwen3:8b")]
