import asyncio
import json
from datetime import UTC, datetime

from atlas.llm.models import ToolCall, ToolFunction
from atlas.tools import CurrentTimeTool, ToolRegistry


def test_current_time_tool_returns_structured_local_time() -> None:
    registry = ToolRegistry(
        [CurrentTimeTool(clock=lambda: datetime(2026, 9, 2, 9, 30, tzinfo=UTC))],
        execution_timeout_seconds=1,
    )

    async def exercise():
        return await registry.execute(
            ToolCall(function=ToolFunction(name="get_current_time", arguments={}))
        )

    result = asyncio.run(exercise())

    assert result.ok is True
    assert result.tool_name == "get_current_time"
    payload = json.loads(result.content)
    assert datetime.fromisoformat(payload["local_time"]) == datetime(
        2026, 9, 2, 9, 30, tzinfo=UTC
    )
    assert payload["weekday"] == datetime.fromisoformat(payload["local_time"]).strftime("%A")
    assert payload["timezone"]


def test_registry_rejects_tool_arguments_outside_the_declared_schema() -> None:
    registry = ToolRegistry([CurrentTimeTool()], execution_timeout_seconds=1)

    async def exercise():
        return await registry.execute(
            ToolCall(
                function=ToolFunction(name="get_current_time", arguments={"timezone": "UTC"})
            )
        )

    result = asyncio.run(exercise())

    assert result.ok is False
    assert result.content.startswith("Invalid arguments:")
