import asyncio
import json

import httpx
import pytest

from atlas.config import Settings
from atlas.llm.models import ToolCall, ToolFunction
from atlas.tools import CurrentWeatherTool, ToolRegistry
from atlas.weather import OpenMeteoWeatherClient, WeatherError, WeatherSnapshot


def _weather_payload() -> dict[str, object]:
    return {
        "utc_offset_seconds": 10800,
        "timezone": "Europe/Istanbul",
        "current": {
            "time": "2026-09-04T19:30",
            "temperature_2m": 24.5,
            "relative_humidity_2m": 62,
            "apparent_temperature": 24.8,
            "precipitation": 0.0,
            "weather_code": 2,
            "wind_speed_10m": 11.2,
            "is_day": 1,
        },
    }


def test_weather_client_uses_fixed_home_coordinates_and_caches_results() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=_weather_payload())

    async def exercise() -> WeatherSnapshot:
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        weather = OpenMeteoWeatherClient(
            Settings(weather_latitude=41.0082, weather_longitude=28.9784),
            client=client,
        )
        first = await weather.current()
        second = await weather.current()
        await client.aclose()
        assert first is second
        return first

    snapshot = asyncio.run(exercise())

    assert len(requests) == 1
    assert requests[0].url.params["latitude"] == "41.0082"
    assert requests[0].url.params["longitude"] == "28.9784"
    assert requests[0].url.params["current"] == (
        "temperature_2m,relative_humidity_2m,apparent_temperature,precipitation,"
        "weather_code,wind_speed_10m,is_day"
    )
    assert snapshot.condition == "partly cloudy"
    assert snapshot.observed_at.isoformat() == "2026-09-04T19:30:00+03:00"


def test_weather_client_returns_a_controlled_error_for_bad_provider_data() -> None:
    async def exercise() -> None:
        client = httpx.AsyncClient(
            transport=httpx.MockTransport(lambda _: httpx.Response(200, json={}))
        )
        weather = OpenMeteoWeatherClient(
            Settings(weather_latitude=41.0082, weather_longitude=28.9784),
            client=client,
        )
        with pytest.raises(WeatherError, match="Weather is unavailable"):
            await weather.current()
        await client.aclose()

    asyncio.run(exercise())


def test_weather_tool_returns_structured_current_conditions() -> None:
    class StaticWeatherReader:
        async def current(self) -> WeatherSnapshot:
            return WeatherSnapshot(
                observed_at="2026-09-04T19:30:00+03:00",
                timezone="Europe/Istanbul",
                condition="partly cloudy",
                weather_code=2,
                temperature_c=24.5,
                apparent_temperature_c=24.8,
                humidity_percent=62,
                precipitation_mm=0,
                wind_speed_kph=11.2,
                is_day=True,
            )

    registry = ToolRegistry([CurrentWeatherTool(StaticWeatherReader())], execution_timeout_seconds=1)

    async def exercise():
        return await registry.execute(
            ToolCall(function=ToolFunction(name="get_current_weather", arguments={}))
        )

    result = asyncio.run(exercise())

    assert result.ok is True
    assert json.loads(result.content)["condition"] == "partly cloudy"


def test_weather_requires_a_complete_home_location() -> None:
    with pytest.raises(ValueError, match="configured together"):
        Settings(weather_latitude=41.0082)
