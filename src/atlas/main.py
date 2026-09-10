"""HTTP entry point and lifecycle management for ATLAS."""

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import structlog
import uvicorn
from fastapi import FastAPI, HTTPException, Query, Request, status
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from atlas.config import Settings, get_settings
from atlas.conversations import (
    ConversationNotFoundError,
    ConversationStore,
    ConversationSummary,
    PersonalMemory,
    StoredConversation,
    StoredEvent,
    StoredToolActivity,
)
from atlas.events import AtlasEvent
from atlas.llm.models import ChatRequest, ChatResponse, ConversationMessage, ToolCall
from atlas.llm.ollama import OllamaProvider
from atlas.llm.provider import LLMError, LLMProvider
from atlas.llm.service import ChatService
from atlas.logging import configure_logging
from atlas.mqtt import MQTTEventAdapter
from atlas.memory import PersonalMemoryWorker
from atlas.obsidian import ObsidianVault
from atlas.tools import (
    CurrentTimeTool,
    CurrentWeatherTool,
    ReadObsidianNoteTool,
    SearchObsidianNotesTool,
    ToolRegistry,
    WriteObsidianNoteTool,
)
from atlas.weather import OpenMeteoWeatherClient

log = structlog.get_logger(__name__)


class HealthStatus(BaseModel):
    status: str
    service: str
    environment: str


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings: Settings = app.state.settings
    # Readiness changes only at lifecycle boundaries, leaving room to initialize
    # databases, workers, and event-bus connections before accepting traffic.
    app.state.conversation_store.initialize()
    app.state.mqtt_task = None
    app.state.memory_task = asyncio.create_task(
        app.state.personal_memory_worker.run(), name="atlas-personal-memory"
    )
    if settings.mqtt_enabled:
        app.state.mqtt_task = asyncio.create_task(app.state.mqtt_adapter.run(), name="atlas-mqtt")
    app.state.ready = True
    log.info("atlas.started", system_event=AtlasEvent(type="system.started").model_dump(mode="json"))
    try:
        yield
    finally:
        app.state.ready = False
        memory_task: asyncio.Task[None] = app.state.memory_task
        memory_task.cancel()
        await asyncio.gather(memory_task, return_exceptions=True)
        mqtt_task: asyncio.Task[None] | None = app.state.mqtt_task
        if mqtt_task is not None:
            mqtt_task.cancel()
            await asyncio.gather(mqtt_task, return_exceptions=True)
        await app.state.llm_provider.aclose()
        weather_client: OpenMeteoWeatherClient | None = app.state.weather_client
        if weather_client is not None:
            await weather_client.aclose()
        # A dedicated shutdown boundary makes later worker cleanup deterministic.
        await asyncio.sleep(0)
        log.info("atlas.stopped", system_event=AtlasEvent(type="system.stopped").model_dump(mode="json"))


def create_app(settings: Settings | None = None, provider: LLMProvider | None = None) -> FastAPI:
    """Build ATLAS without starting the server, enabling reliable tests."""
    settings = settings or get_settings()
    configure_logging(settings.log_level)
    # The factory keeps construction separate from process startup; tests and
    # future workers can create isolated app instances with explicit settings.
    app = FastAPI(title=settings.app_name, version="0.1.0", lifespan=lifespan)
    app.state.settings = settings
    app.state.ready = False
    app.state.conversation_store = ConversationStore(settings.database_path)
    app.state.mqtt_adapter = MQTTEventAdapter(settings, app.state.conversation_store.record_event)
    app.state.llm_provider = provider or OllamaProvider(settings)
    app.state.personal_memory_worker = PersonalMemoryWorker(
        settings, app.state.llm_provider, app.state.conversation_store
    )
    registered_tools = [CurrentTimeTool()]
    app.state.weather_client = OpenMeteoWeatherClient(settings)
    registered_tools.append(CurrentWeatherTool(app.state.weather_client))
    app.state.obsidian_vault = None
    if settings.obsidian_vault_path.strip():
        app.state.obsidian_vault = ObsidianVault(settings.obsidian_vault_path)
        registered_tools.extend(
            [
                SearchObsidianNotesTool(app.state.obsidian_vault),
                ReadObsidianNoteTool(app.state.obsidian_vault),
            ]
        )
        if settings.obsidian_write_enabled:
            registered_tools.append(WriteObsidianNoteTool(app.state.obsidian_vault))
    app.state.chat_service = ChatService(
        settings,
        app.state.llm_provider,
        tool_registry=ToolRegistry(
            registered_tools,
            execution_timeout_seconds=settings.tool_execution_timeout_seconds,
        ),
    )

    def conversation_messages(request: ChatRequest) -> tuple[list[ConversationMessage], str | None]:
        """Use persisted history only when the client explicitly names a thread."""
        if request.conversation_id is None:
            return list(request.messages), None
        if len(request.messages) != 1:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="persisted conversations accept one new user message per request",
            )
        try:
            app.state.conversation_store.get_conversation(request.conversation_id)
        except ConversationNotFoundError as error:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found.") from error
        user_message = request.messages[0]
        app.state.conversation_store.append_message(
            request.conversation_id,
            role="user",
            content=user_message.content,
        )
        conversation = app.state.conversation_store.get_conversation(request.conversation_id)
        # A provider can occasionally finish a turn with no user-facing text.
        # Keep that audit record, but do not send an invalid blank assistant
        # message back through the client-facing conversation schema.
        return [
            message.as_conversation_message()
            for message in conversation.messages
            if message.content
        ], request.conversation_id

    def personal_memory_facts() -> list[str]:
        """Keep the small, reviewed profile useful to ATLAS without exposing raw history."""
        return [memory.fact for memory in app.state.conversation_store.list_personal_memories(limit=40)]

    @app.exception_handler(LLMError)
    async def llm_error_handler(_: Request, error: LLMError) -> JSONResponse:
        log.warning("chat.failed", error_type=type(error).__name__)
        return JSONResponse(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, content={"detail": str(error)})

    @app.get("/health/live", response_model=HealthStatus, tags=["health"])
    async def liveness() -> HealthStatus:
        return HealthStatus(status="ok", service=settings.app_name, environment=settings.environment)

    @app.get("/health/ready", response_model=HealthStatus, tags=["health"])
    async def readiness() -> HealthStatus:
        if not app.state.ready:
            return HealthStatus(status="starting", service=settings.app_name, environment=settings.environment)
        return HealthStatus(status="ok", service=settings.app_name, environment=settings.environment)

    @app.get("/events", response_model=list[StoredEvent], tags=["events"])
    async def list_events(limit: int = Query(default=50, ge=1, le=200)) -> list[StoredEvent]:
        return app.state.conversation_store.list_events(limit=limit)

    @app.get("/memories", response_model=list[PersonalMemory], tags=["memory"])
    async def list_personal_memories(limit: int = Query(default=100, ge=1, le=200)) -> list[PersonalMemory]:
        return app.state.conversation_store.list_personal_memories(limit=limit)

    @app.post("/conversations", response_model=ConversationSummary, status_code=status.HTTP_201_CREATED, tags=["conversations"])
    async def create_conversation() -> ConversationSummary:
        return app.state.conversation_store.create_conversation()

    @app.get("/conversations", response_model=list[ConversationSummary], tags=["conversations"])
    async def list_conversations() -> list[ConversationSummary]:
        return app.state.conversation_store.list_conversations()

    @app.get("/conversations/{conversation_id}", response_model=StoredConversation, tags=["conversations"])
    async def get_conversation(conversation_id: str) -> StoredConversation:
        try:
            return app.state.conversation_store.get_conversation(conversation_id)
        except ConversationNotFoundError as error:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found.") from error

    @app.post("/chat", response_model=ChatResponse, tags=["chat"])
    async def chat(request: ChatRequest) -> ChatResponse:
        messages, conversation_id = conversation_messages(request)
        completion = await app.state.chat_service.reply(
            messages, request.brain, personal_memory=personal_memory_facts()
        )
        if conversation_id is not None:
            app.state.conversation_store.append_message(
                conversation_id,
                role="assistant",
                content=completion.message.content,
                brain=request.brain,
                model=completion.model,
            )
        return ChatResponse(
            content=completion.message.content,
            brain=request.brain,
            model=completion.model,
            usage=completion.usage,
        )

    @app.post("/chat/stream", tags=["chat"])
    async def stream_chat(request: ChatRequest) -> StreamingResponse:
        messages, conversation_id = conversation_messages(request)

        async def events() -> AsyncIterator[str]:
            content: list[str] = []
            tool_activity: list[StoredToolActivity] = []
            model = app.state.chat_service.model_for(request.brain)
            try:
                async for event in app.state.chat_service.stream(
                    messages, request.brain, personal_memory=personal_memory_facts()
                ):
                    if event.type == "token":
                        content.append(event.content)
                    elif event.type == "tool_call":
                        tool_activity.extend(_tool_activity_from_calls(event.tool_calls))
                    elif event.type == "tool_result":
                        _apply_tool_results(tool_activity, event.tool_results)
                    elif event.type == "done" and event.model is not None:
                        model = event.model
                    yield _sse(event.type, event.model_dump(mode="json", exclude_none=True))
                if conversation_id is not None:
                    app.state.conversation_store.append_message(
                        conversation_id,
                        role="assistant",
                        content="".join(content),
                        brain=request.brain,
                        model=model,
                        tool_activity=tool_activity,
                    )
            except LLMError as error:
                log.warning("chat.stream_failed", error_type=type(error).__name__)
                yield _sse("error", {"detail": str(error)})

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return app


app = create_app()


def run() -> None:
    settings = get_settings()
    uvicorn.run("atlas.main:app", host=settings.host, port=settings.port, log_config=None)


def _sse(event: str, data: dict[str, object]) -> str:
    """Encode one server-sent event without leaking implementation details."""
    return f"event: {event}\ndata: {json.dumps(data, separators=(',', ':'))}\n\n"


def _tool_activity_from_calls(tool_calls: list[ToolCall]) -> list[StoredToolActivity]:
    """Convert provider-neutral requests into the durable audit record."""
    return [
        StoredToolActivity(
            tool_call_id=tool_call.id,
            tool_name=tool_call.function.name,
            arguments=tool_call.function.arguments,
        )
        for tool_call in tool_calls
    ]


def _apply_tool_results(activity: list[StoredToolActivity], results: list[dict[str, object]]) -> None:
    """Attach controlled tool outcomes to their original requests."""
    for result in results:
        tool_call_id = result.get("tool_call_id")
        tool_name = result.get("tool_name")
        match = next(
            (
                item
                for item in activity
                if item.tool_call_id == tool_call_id
                or (tool_call_id is None and item.tool_name == tool_name and item.ok is None)
            ),
            None,
        )
        if match is None:
            continue
        ok = result.get("ok")
        content = result.get("content")
        match.ok = ok if isinstance(ok, bool) else False
        match.content = content if isinstance(content, str) else "Tool execution returned an invalid result."
