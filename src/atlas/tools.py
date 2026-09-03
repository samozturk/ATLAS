"""Application-owned tool definitions and safe execution for Phase 3."""

import asyncio
import json
from collections.abc import Callable, Iterable
from datetime import datetime
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from atlas.llm.models import ChatMessage, MessageRole, ToolCall, ToolDefinition


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
        except Exception:
            # The model receives a controlled failure message; implementation
            # details remain in server logs once observability is added.
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
