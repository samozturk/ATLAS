"""Application-owned tool definitions and safe execution for Phase 3."""

import asyncio
import json
from collections.abc import Callable, Iterable
from datetime import datetime
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError
import structlog

from atlas.llm.models import ChatMessage, MessageRole, ToolCall, ToolDefinition
from atlas.weather import WeatherError, WeatherSnapshot

log = structlog.get_logger(__name__)


class ToolInput(BaseModel):
    """Base input model; tools opt into arguments through typed fields."""

    model_config = ConfigDict(extra="forbid")


class ToolExecution(BaseModel):
    """An audited, model-readable result of one requested tool invocation."""

    tool_call_id: str | None = None
    tool_name: str
    ok: bool
    content: str

    def as_message(self) -> ChatMessage:
        try:
            result: Any = json.loads(self.content)
        except json.JSONDecodeError:
            result = self.content
        return ChatMessage(
            role=MessageRole.TOOL,
            tool_name=self.tool_name,
            content=json.dumps(
                {"ok": self.ok, "result": result},
                separators=(",", ":"),
            ),
        )


class ToolExecutionFailure(Exception):
    """A known, safe error that should be returned to the model and user."""


class AtlasTool(Protocol):
    """Standard interface for tools that ATLAS, rather than the model, owns."""

    name: str
    description: str
    input_model: type[ToolInput]

    @property
    def definition(self) -> ToolDefinition: ...

    async def execute(self, arguments: ToolInput) -> str: ...


class CurrentTimeInput(ToolInput):
    """The local clock needs no caller-supplied arguments."""


class CurrentTimeTool:
    """Return the local clock on the ATLAS host; this tool never changes state."""

    name = "get_current_time"
    description = "Get the current local date, time, weekday, and timezone at the ATLAS home."
    input_model = CurrentTimeInput

    def __init__(self, clock: Callable[[], datetime] = datetime.now) -> None:
        self._clock = clock

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            function={
                "name": self.name,
                "description": self.description,
                "parameters": self.input_model.model_json_schema(),
            }
        )

    async def execute(self, arguments: CurrentTimeInput) -> str:
        del arguments
        now = self._clock().astimezone()
        return json.dumps(
            {
                "local_time": now.isoformat(),
                "weekday": now.strftime("%A"),
                "timezone": now.tzname(),
            },
            separators=(",", ":"),
        )


class CurrentWeatherInput(ToolInput):
    """An optional user-requested place; omitted uses the configured home."""

    location: str | None = Field(default=None, min_length=2, max_length=120)


class WeatherReader(Protocol):
    """Read the current conditions from an approved weather adapter."""

    async def current(self, location: str | None = None) -> WeatherSnapshot: ...


class CurrentWeatherTool:
    """Return current conditions for a requested place or ATLAS's home."""

    name = "get_current_weather"
    description = (
        "Get current weather. Omit location for the configured ATLAS home in Waalwijk; "
        "provide location only when the user asks about a different place."
    )
    input_model = CurrentWeatherInput

    def __init__(self, reader: WeatherReader) -> None:
        self._reader = reader

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            function={
                "name": self.name,
                "description": self.description,
                "parameters": self.input_model.model_json_schema(),
            }
        )

    async def execute(self, arguments: CurrentWeatherInput) -> str:
        try:
            snapshot = await self._reader.current(arguments.location)
        except WeatherError as error:
            raise ToolExecutionFailure(str(error)) from error
        return json.dumps(snapshot.model_dump(mode="json"), separators=(",", ":"))


class ToolRegistry:
    """Validate and execute only the tools explicitly registered by ATLAS."""

    def __init__(self, tools: Iterable[AtlasTool], *, execution_timeout_seconds: float) -> None:
        registered_tools = list(tools)
        self._tools = {tool.name: tool for tool in registered_tools}
        if not self._tools:
            raise ValueError("at least one tool must be registered")
        if len(self._tools) != len(registered_tools):
            raise ValueError("tool names must be unique")
        self._execution_timeout_seconds = execution_timeout_seconds

    @property
    def definitions(self) -> list[ToolDefinition]:
        return [tool.definition for tool in self._tools.values()]

    async def execute(self, tool_call: ToolCall) -> ToolExecution:
        tool_name = tool_call.function.name
        tool = self._tools.get(tool_name)
        if tool is None:
            return ToolExecution(
                tool_call_id=tool_call.id,
                tool_name=tool_name,
                ok=False,
                content=f"Unknown tool: {tool_name}",
            )

        try:
            arguments = tool.input_model.model_validate(tool_call.function.arguments)
        except ValidationError as error:
            return ToolExecution(
                tool_call_id=tool_call.id,
                tool_name=tool_name,
                ok=False,
                content=f"Invalid arguments: {error.errors(include_url=False)}",
            )

        try:
            async with asyncio.timeout(self._execution_timeout_seconds):
                content = await tool.execute(arguments)
        except TimeoutError:
            return ToolExecution(
                tool_call_id=tool_call.id,
                tool_name=tool_name,
                ok=False,
                content="Tool execution timed out.",
            )
        except ToolExecutionFailure as error:
            log.warning(
                "tool.execution_failed",
                tool_name=tool_name,
                error_type=type(error.__cause__).__name__,
                detail=str(error),
            )
            return ToolExecution(
                tool_call_id=tool_call.id,
                tool_name=tool_name,
                ok=False,
                content=str(error),
            )
        except Exception as error:
            log.exception(
                "tool.execution_failed",
                tool_name=tool_name,
                error_type=type(error).__name__,
            )
            return ToolExecution(
                tool_call_id=tool_call.id,
                tool_name=tool_name,
                ok=False,
                content="Tool execution failed.",
            )

        return ToolExecution(
            tool_call_id=tool_call.id,
            tool_name=tool_name,
            ok=True,
            content=content,
        )
