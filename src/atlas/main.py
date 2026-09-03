"""HTTP entry point and lifecycle management for ATLAS."""

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import structlog
import uvicorn
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from atlas.config import Settings, get_settings
from atlas.events import AtlasEvent
from atlas.llm.models import ChatRequest, ChatResponse
from atlas.llm.ollama import OllamaProvider
from atlas.llm.provider import LLMError, LLMProvider
from atlas.llm.service import ChatService
from atlas.logging import configure_logging

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
    app.state.ready = True
    log.info("atlas.started", system_event=AtlasEvent(type="system.started").model_dump(mode="json"))
    try:
        yield
    finally:
        app.state.ready = False
        await app.state.llm_provider.aclose()
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
    app.state.llm_provider = provider or OllamaProvider(settings)
    app.state.chat_service = ChatService(settings, app.state.llm_provider)

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

    @app.post("/chat", response_model=ChatResponse, tags=["chat"])
    async def chat(request: ChatRequest) -> ChatResponse:
        completion = await app.state.chat_service.reply(request.messages, request.brain)
        return ChatResponse(
            content=completion.message.content,
            brain=request.brain,
            model=completion.model,
            usage=completion.usage,
        )

    @app.post("/chat/stream", tags=["chat"])
    async def stream_chat(request: ChatRequest) -> StreamingResponse:
        async def events() -> AsyncIterator[str]:
            try:
                async for event in app.state.chat_service.stream(request.messages, request.brain):
                    yield _sse(event.type, event.model_dump(mode="json", exclude_none=True))
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
