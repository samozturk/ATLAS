"""Ollama adapter with bounded retries and native streaming support."""

import asyncio
import json
from collections.abc import AsyncIterator, Sequence
from typing import Any

import httpx
import structlog

from atlas.config import Settings
from atlas.llm.models import (
    ChatCompletion,
    ChatMessage,
    ChatStreamEvent,
    MessageRole,
    ToolCall,
    ToolDefinition,
    ToolFunction,
    Usage,
)
from atlas.llm.provider import LLMResponseError, LLMUnavailableError

log = structlog.get_logger(__name__)


class OllamaProvider:
    """Translate ATLAS's LLM contract to Ollama's local chat API."""

    def __init__(self, settings: Settings, client: httpx.AsyncClient | None = None) -> None:
        self._settings = settings
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            base_url=settings.ollama_base_url.rstrip("/"),
            timeout=httpx.Timeout(
                timeout=settings.llm_request_timeout_seconds,
                connect=settings.llm_connect_timeout_seconds,
            ),
        )

    async def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        model: str,
        tools: Sequence[ToolDefinition] = (),
    ) -> ChatCompletion:
        payload = self._payload(messages, model=model, tools=tools, stream=False)
        body = await self._post_with_retries(payload)
        return self._completion_from_body(body, requested_model=model)

    async def stream(
        self,
        messages: Sequence[ChatMessage],
        *,
        model: str,
        tools: Sequence[ToolDefinition] = (),
    ) -> AsyncIterator[ChatStreamEvent]:
        """Stream Ollama NDJSON, retrying only before the first emitted item."""
        payload = self._payload(messages, model=model, tools=tools, stream=True)
        attempt = 0
        while True:
            emitted = False
            try:
                async with self._client.stream("POST", "/api/chat", json=payload) as response:
                    if response.is_error:
                        detail = await self._error_detail(response)
                        if response.status_code < 500:
                            raise LLMResponseError(detail)
                        raise _RetryableProviderError(detail)

                    async for line in response.aiter_lines():
                        if not line:
                            continue
                        try:
                            chunk = json.loads(line)
                        except json.JSONDecodeError as error:
                            raise LLMResponseError("Ollama returned invalid streaming JSON") from error
                        emitted = True
                        yield self._stream_event_from_chunk(chunk, requested_model=model)
                    return
            except LLMResponseError:
                raise
            except (_RetryableProviderError, httpx.TransportError, httpx.TimeoutException) as error:
                if emitted or attempt >= self._settings.llm_max_retries:
                    raise LLMUnavailableError("Ollama is unavailable; try again shortly") from error
                await self._retry(attempt, error)
                attempt += 1

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    def _payload(
        self,
        messages: Sequence[ChatMessage],
        *,
        model: str,
        tools: Sequence[ToolDefinition],
        stream: bool,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model,
            "messages": [message.as_ollama() for message in messages],
            "stream": stream,
        }
        if tools:
            payload["tools"] = [tool.model_dump(mode="json") for tool in tools]
        return payload

    async def _post_with_retries(self, payload: dict[str, Any]) -> dict[str, Any]:
        for attempt in range(self._settings.llm_max_retries + 1):
            try:
                response = await self._client.post("/api/chat", json=payload)
                if response.is_error:
                    detail = await self._error_detail(response)
                    if response.status_code < 500:
                        raise LLMResponseError(detail)
                    raise _RetryableProviderError(detail)
                try:
                    body = response.json()
                except json.JSONDecodeError as error:
                    raise LLMResponseError("Ollama returned invalid response JSON") from error
                if not isinstance(body, dict):
                    raise LLMResponseError("Ollama returned an invalid response body")
                return body
            except LLMResponseError:
                raise
            except (_RetryableProviderError, httpx.TransportError, httpx.TimeoutException) as error:
                if attempt >= self._settings.llm_max_retries:
                    raise LLMUnavailableError("Ollama is unavailable; try again shortly") from error
                await self._retry(attempt, error)
        raise AssertionError("retry loop must return or raise")

    async def _retry(self, attempt: int, error: Exception) -> None:
        delay = self._settings.llm_retry_backoff_seconds * (2**attempt)
        log.warning(
            "llm.request_retry",
            provider="ollama",
            attempt=attempt + 1,
            retry_delay_seconds=delay,
            error_type=type(error).__name__,
        )
        if delay:
            await asyncio.sleep(delay)

    async def _error_detail(self, response: httpx.Response) -> str:
        try:
            body = response.json()
        except json.JSONDecodeError:
            body = None
        if isinstance(body, dict) and isinstance(body.get("error"), str):
            return f"Ollama request failed: {body['error']}"
        return f"Ollama request failed with HTTP {response.status_code}"

    def _completion_from_body(self, body: dict[str, Any], *, requested_model: str) -> ChatCompletion:
        message_data = body.get("message")
        if not isinstance(message_data, dict):
            raise LLMResponseError("Ollama response did not contain an assistant message")
        model = self._response_model(body, requested_model)
        return ChatCompletion(
            message=self._message_from_data(message_data),
            model=model,
            usage=self._usage_from_data(body),
        )

    def _stream_event_from_chunk(self, chunk: dict[str, Any], *, requested_model: str) -> ChatStreamEvent:
        if chunk.get("done"):
            return ChatStreamEvent(
                type="done",
                model=self._response_model(chunk, requested_model),
                usage=self._usage_from_data(chunk),
            )

        message = chunk.get("message")
        if not isinstance(message, dict):
            raise LLMResponseError("Ollama streaming response did not contain a message")
        tool_calls = self._tool_calls_from_data(message.get("tool_calls"))
        if tool_calls:
            return ChatStreamEvent(type="tool_call", tool_calls=tool_calls)
        thinking = message.get("thinking")
        if thinking:
            if not isinstance(thinking, str):
                raise LLMResponseError("Ollama returned invalid thinking content")
            return ChatStreamEvent(type="thinking", content=thinking)
        content = message.get("content", "")
        if not isinstance(content, str):
            raise LLMResponseError("Ollama returned invalid message content")
        return ChatStreamEvent(type="token", content=content)

    def _message_from_data(self, data: dict[str, Any]) -> ChatMessage:
        raw_role = data.get("role", "assistant")
        try:
            role = MessageRole(raw_role)
        except ValueError as error:
            raise LLMResponseError("Ollama returned an unknown message role") from error
        content = data.get("content", "")
        if not isinstance(content, str):
            raise LLMResponseError("Ollama returned invalid message content")
        thinking = data.get("thinking")
        if thinking is not None and not isinstance(thinking, str):
            raise LLMResponseError("Ollama returned invalid thinking content")
        tool_name = data.get("tool_name")
        if tool_name is not None and not isinstance(tool_name, str):
            raise LLMResponseError("Ollama returned invalid tool name")
        return ChatMessage(
            role=role,
            content=content,
            thinking=thinking,
            tool_calls=self._tool_calls_from_data(data.get("tool_calls")),
            tool_name=tool_name,
        )

    def _tool_calls_from_data(self, raw_calls: Any) -> list[ToolCall]:
        if raw_calls is None:
            return []
        if not isinstance(raw_calls, list):
            raise LLMResponseError("Ollama returned invalid tool calls")
        calls: list[ToolCall] = []
        for raw_call in raw_calls:
            if not isinstance(raw_call, dict) or not isinstance(raw_call.get("function"), dict):
                raise LLMResponseError("Ollama returned invalid tool call data")
            function = raw_call["function"]
            name = function.get("name")
            arguments = function.get("arguments", {})
            if not isinstance(name, str):
                raise LLMResponseError("Ollama returned a tool call without a name")
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError as error:
                    raise LLMResponseError("Ollama returned invalid tool arguments") from error
            if not isinstance(arguments, dict):
                raise LLMResponseError("Ollama returned invalid tool arguments")
            calls.append(
                ToolCall(
                    id=raw_call.get("id"),
                    function=ToolFunction(name=name, arguments=arguments),
                )
            )
        return calls

    @staticmethod
    def _usage_from_data(data: dict[str, Any]) -> Usage:
        prompt_tokens = data.get("prompt_eval_count")
        completion_tokens = data.get("eval_count")
        return Usage(
            prompt_tokens=prompt_tokens if isinstance(prompt_tokens, int) and prompt_tokens >= 0 else None,
            completion_tokens=completion_tokens
            if isinstance(completion_tokens, int) and completion_tokens >= 0
            else None,
        )

    @staticmethod
    def _response_model(data: dict[str, Any], requested_model: str) -> str:
        model = data.get("model")
        return model if isinstance(model, str) and model else requested_model


class _RetryableProviderError(Exception):
    """Internal marker for transport and server errors that can be retried."""
