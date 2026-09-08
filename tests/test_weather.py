import asyncio
import json

import httpx
import pytest

from atlas.config import Settings
from atlas.llm.models import ToolCall, ToolFunction
from atlas.tools import CurrentWeatherTool, ToolRegistry
from atlas.weather import (
    OpenMeteoWeatherClient,
    WeatherLocationNotFoundError,
    WeatherSnapshot,
)


def _weather_payload() -> dict[str, object]:
    return {
        "utc_offset_seconds": 7200,
        "timezone": "Europe/Amsterdam",
        "current": {
            "time": "2026-09-08T17:30",
            "temperature_2m": 18.5,
            "relative_humidity_2m": 62,
            "apparent_temperature": 18.3,
            "precipitation": 0.0,
            "weather_code": 2,
            "wind_speed_10m": 11.2,
            "is_day": 1,
        },
    }


def _geocoding_payload(name: str, latitude: float, longitude: float) -> dict[str, object]:
    return {
        "results": [
            {
                "name": name,
                "admin1": "North Brabant" if name == "Waalwijk" else None,
                "country": "Netherlands",
                "latitude": latitude,
                "longitude": longitude,
            }
        ]
    }


def test_weather_client_geocodes_the_default_and_requested_locations_then_caches() -> None:
    geocoding_requests: list[httpx.Request] = []
    forecast_requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "geocoding-api.open-meteo.com":
            geocoding_requests.append(request)
            if request.url.params["name"] == "Waalwijk, Netherlands":
                return httpx.Response(200, json=_geocoding_payload("Waalwijk", 51.6825, 5.0708))
            return httpx.Response(200, json=_geocoding_payload("Amsterdam", 52.3676, 4.9041))
        forecast_requests.append(request)
        return httpx.Response(200, json=_weather_payload())

    async def exercise() -> tuple[WeatherSnapshot, WeatherSnapshot]:
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        weather = OpenMeteoWeatherClient(Settings(), client=client)
        default = await weather.current()
        assert await weather.current() is default
        requested = await weather.current("Amsterdam, Netherlands")
        await client.aclose()
        return default, requested

    default, requested = asyncio.run(exercise())

    assert [request.url.params["name"] for request in geocoding_requests] == [
        "Waalwijk, Netherlands",
        "Amsterdam, Netherlands",
    ]
    assert len(forecast_requests) == 2
    assert forecast_requests[0].url.params["latitude"] == "51.6825"
    assert forecast_requests[1].url.params["longitude"] == "4.9041"
    assert default.location == "Waalwijk, North Brabant, Netherlands"
    assert requested.location == "Amsterdam, Netherlands"
    assert requested.condition == "partly cloudy"
    assert requested.observed_at.isoformat() == "2026-09-08T17:30:00+02:00"


def test_weather_client_returns_a_clear_error_when_a_location_is_not_found() -> None:
    async def exercise() -> None:
        client = httpx.AsyncClient(
            transport=httpx.MockTransport(lambda _: httpx.Response(200, json={"results": []}))
        )
        weather = OpenMeteoWeatherClient(Settings(), client=client)
        with pytest.raises(WeatherLocationNotFoundError, match="could not find"):
            await weather.current("Nowhereville")
        await client.aclose()

    asyncio.run(exercise())


def test_weather_tool_forwards_an_optional_user_requested_location() -> None:
    class StaticWeatherReader:
        def __init__(self) -> None:
            self.requested_location: str | None = None

        async def current(self, location: str | None = None) -> WeatherSnapshot:
            self.requested_location = location
            return WeatherSnapshot(
                location="Amsterdam, Netherlands",
                latitude=52.3676,
                longitude=4.9041,
                observed_at="2026-09-08T17:30:00+02:00",
                timezone="Europe/Amsterdam",
                condition="partly cloudy",
                weather_code=2,
                temperature_c=18.5,
                apparent_temperature_c=18.3,
                humidity_percent=62,
                precipitation_mm=0,
                wind_speed_kph=11.2,
                is_day=True,
            )

    reader = StaticWeatherReader()
    registry = ToolRegistry([CurrentWeatherTool(reader)], execution_timeout_seconds=1)

    async def exercise():
        return await registry.execute(
            ToolCall(
                function=ToolFunction(
                    name="get_current_weather", arguments={"location": "Amsterdam, Netherlands"}
                )
            )
        )

    result = asyncio.run(exercise())

    assert result.ok is True
    assert reader.requested_location == "Amsterdam, Netherlands"
    assert json.loads(result.content)["location"] == "Amsterdam, Netherlands"


def test_weather_tool_returns_a_safe_location_lookup_error() -> None:
    class MissingWeatherReader:
        async def current(self, location: str | None = None) -> WeatherSnapshot:
            del location
            raise WeatherLocationNotFoundError("ATLAS could not find a weather location for 'Nowhereville'.")

    registry = ToolRegistry([CurrentWeatherTool(MissingWeatherReader())], execution_timeout_seconds=1)

    async def exercise():
        return await registry.execute(
            ToolCall(function=ToolFunction(name="get_current_weather", arguments={}))
        )

    result = asyncio.run(exercise())

    assert result.ok is False
    assert result.content == "ATLAS could not find a weather location for 'Nowhereville'."
