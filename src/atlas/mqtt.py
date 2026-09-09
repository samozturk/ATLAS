"""Read-only MQTT intake that translates approved broker messages into ATLAS events."""

import asyncio
import json
from collections.abc import Callable
from typing import Any

import aiomqtt
import structlog

from atlas.config import Settings
from atlas.events import AtlasEvent

log = structlog.get_logger(__name__)


class MQTTEventAdapter:
    """Subscribe to the ATLAS event namespace without publishing device commands."""

    def __init__(self, settings: Settings, record_event: Callable[[AtlasEvent], None]) -> None:
        self._settings = settings
        self._record_event = record_event
        self._topic_prefix = settings.mqtt_event_topic_prefix.strip("/")

    async def run(self) -> None:
        """Reconnect with bounded delay until the application cancels this task."""
        if not self._settings.mqtt_enabled:
            return
        if not self._settings.mqtt_host.strip():
            log.warning("mqtt.disabled", reason="missing_host")
            return
        while True:
            try:
                await self._consume()
            except asyncio.CancelledError:
                raise
            except aiomqtt.MqttError as error:
                log.warning(
                    "mqtt.disconnected",
                    error_type=type(error).__name__,
                    retry_delay_seconds=self._settings.mqtt_reconnect_delay_seconds,
                )
                await asyncio.sleep(self._settings.mqtt_reconnect_delay_seconds)

    async def _consume(self) -> None:
        password = (
            self._settings.mqtt_password.get_secret_value()
            if self._settings.mqtt_password is not None
            else None
        )
        async with aiomqtt.Client(
            hostname=self._settings.mqtt_host,
            port=self._settings.mqtt_port,
            username=self._settings.mqtt_username or None,
            password=password,
            identifier=self._settings.mqtt_client_id,
        ) as client:
            await client.subscribe(self._settings.mqtt_topic_filter)
            log.info("mqtt.connected", topic_filter=self._settings.mqtt_topic_filter)
            async for message in client.messages:
                event = self.event_from_message(str(message.topic), bytes(message.payload))
                if event is not None:
                    self._record_event(event)
                    log.info("mqtt.event_received", event_type=event.type, topic=str(message.topic))

    def event_from_message(self, topic: str, payload: bytes) -> AtlasEvent | None:
        """Accept only the configured namespace and bound untrusted MQTT payloads."""
        prefix = f"{self._topic_prefix}/"
        if not self._topic_prefix or not topic.startswith(prefix):
            log.warning("mqtt.message_rejected", topic=topic, reason="outside_event_namespace")
            return None
        if len(payload) > self._settings.mqtt_max_payload_bytes:
            log.warning("mqtt.message_rejected", topic=topic, reason="payload_too_large")
            return None
        event_type = topic.removeprefix(prefix).strip("/")
        if not event_type:
            log.warning("mqtt.message_rejected", topic=topic, reason="missing_event_type")
            return None
        return AtlasEvent(
            type=event_type,
            source="mqtt",
            payload={"topic": topic, "data": _decode_payload(payload)},
        )


def _decode_payload(payload: bytes) -> Any:
    """Prefer structured JSON, falling back to UTF-8 text or a lossless hex form."""
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError:
        return {"encoding": "hex", "value": payload.hex()}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text
