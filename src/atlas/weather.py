"""Opt-in, cached current-weather integration for ATLAS's configured home."""

import asyncio
from datetime import datetime, timedelta, timezone
from time import monotonic
from typing import Any, Callable

import httpx
from pydantic import BaseModel

from atlas.config import Settings


class WeatherError(Exception):
    """An unavailable or malformed weather response safe to handle as a tool failure."""


class WeatherSnapshot(BaseModel):
    """Normalized current conditions returned to the tool layer."""

    observed_at: datetime
    timezone: str
    condition: str
    weather_code: int
    temperature_c: float
    apparent_temperature_c: float
    humidity_percent: float
    precipitation_mm: float
    wind_speed_kph: float
    is_day: bool
    source: str = "open-meteo"


_WEATHER_CONDITIONS = {
    0: "clear sky",
    1: "mainly clear",
    2: "partly cloudy",
    3: "overcast",
    45: "fog",
    48: "rime fog",
    51: "light drizzle",
    53: "moderate drizzle",
    55: "dense drizzle",
    56: "light freezing drizzle",
    57: "dense freezing drizzle",
    61: "slight rain",
    63: "moderate rain",
    65: "heavy rain",
    66: "light freezing rain",
    67: "heavy freezing rain",
    71: "slight snow fall",
    73: "moderate snow fall",
    75: "heavy snow fall",
    77: "snow grains",
    80: "slight rain showers",
    81: "moderate rain showers",
    82: "violent rain showers",
    85: "slight snow showers",
    86: "heavy snow showers",
    95: "thunderstorm",
    96: "thunderstorm with slight hail",
    99: "thunderstorm with heavy hail",
}


class OpenMeteoWeatherClient:
    """Fetch and cache current conditions for the configured home coordinates."""

    _endpoint = "https://api.open-meteo.com/v1/forecast"
    _current_variables = ",".join(
        (
            "temperature_2m",
            "relative_humidity_2m",
            "apparent_temperature",
            "precipitation",
            "weather_code",
            "wind_speed_10m",
            "is_day",
        )
    )

    def __init__(
        self,
        settings: Settings,
        client: httpx.AsyncClient | None = None,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        if not settings.weather_enabled:
            raise ValueError("weather requires configured home coordinates")
        self._latitude = settings.weather_latitude
        self._longitude = settings.weather_longitude
        self._cache_ttl_seconds = settings.weather_cache_ttl_seconds
        self._clock = clock
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            timeout=httpx.Timeout(settings.weather_request_timeout_seconds)
        )
        self._cache: WeatherSnapshot | None = None
        self._cache_expires_at = 0.0
        self._cache_lock = asyncio.Lock()

    async def current(self) -> WeatherSnapshot:
        """Return cached data when fresh, otherwise issue one provider request."""
        if self._cache is not None and self._clock() < self._cache_expires_at:
            return self._cache

        async with self._cache_lock:
            if self._cache is not None and self._clock() < self._cache_expires_at:
                return self._cache
            snapshot = await self._fetch_current()
            self._cache = snapshot
            self._cache_expires_at = self._clock() + self._cache_ttl_seconds
            return snapshot

    async def aclose(self) -> None:
        """Close the owned HTTP client during application shutdown."""
        if self._owns_client:
            await self._client.aclose()

    async def _fetch_current(self) -> WeatherSnapshot:
        try:
            response = await self._client.get(
                self._endpoint,
                params={
                    "latitude": self._latitude,
                    "longitude": self._longitude,
                    "current": self._current_variables,
                    "timezone": "auto",
                    "temperature_unit": "celsius",
                    "wind_speed_unit": "kmh",
                    "precipitation_unit": "mm",
                },
            )
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict):
                raise ValueError("weather response is not an object")
            return self._snapshot_from_payload(payload)
        except (httpx.HTTPError, TypeError, ValueError) as error:
            raise WeatherError("Weather is unavailable; try again shortly.") from error

    @staticmethod
    def _snapshot_from_payload(payload: dict[str, Any]) -> WeatherSnapshot:
        current = payload.get("current")
        if not isinstance(current, dict):
            raise ValueError("weather response has no current conditions")

        observed_at = _observed_at(current.get("time"), payload.get("utc_offset_seconds"))
        weather_code = _integer(current.get("weather_code"), "weather_code")
        is_day = _integer(current.get("is_day"), "is_day")
        if is_day not in (0, 1):
            raise ValueError("weather response has invalid is_day")

        provider_timezone = payload.get("timezone")
        if not isinstance(provider_timezone, str) or not provider_timezone:
            raise ValueError("weather response has invalid timezone")

        return WeatherSnapshot(
            observed_at=observed_at,
            timezone=provider_timezone,
            condition=_WEATHER_CONDITIONS.get(weather_code, "unknown conditions"),
            weather_code=weather_code,
            temperature_c=_number(current.get("temperature_2m"), "temperature_2m"),
            apparent_temperature_c=_number(
                current.get("apparent_temperature"), "apparent_temperature"
            ),
            humidity_percent=_number(
                current.get("relative_humidity_2m"), "relative_humidity_2m"
            ),
            precipitation_mm=_number(current.get("precipitation"), "precipitation"),
            wind_speed_kph=_number(current.get("wind_speed_10m"), "wind_speed_10m"),
            is_day=bool(is_day),
        )


def _observed_at(raw_time: object, raw_offset_seconds: object) -> datetime:
    if not isinstance(raw_time, str):
        raise ValueError("weather response has invalid time")
    observed_at = datetime.fromisoformat(raw_time)
    if observed_at.tzinfo is not None:
        return observed_at
    offset_seconds = _integer(raw_offset_seconds, "utc_offset_seconds")
    return observed_at.replace(tzinfo=timezone(timedelta(seconds=offset_seconds)))


def _integer(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"weather response has invalid {field}")
    return value


def _number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"weather response has invalid {field}")
    return float(value)
