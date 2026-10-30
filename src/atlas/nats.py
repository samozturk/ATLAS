"""Local NATS transport for normalized ATLAS events."""

import asyncio
from collections.abc import Callable

import nats
import structlog

from atlas.config import Settings
from atlas.events import AtlasEvent

log = structlog.get_logger(__name__)


class NATSEventBus:
    """Publish and consume one broker-neutral event envelope on local NATS."""

    def __init__(self, settings: Settings, receive_event: Callable[[AtlasEvent], None]) -> None:
        self._settings = settings
        self._receive_event = receive_event
        self._subject_prefix = settings.nats_event_subject_prefix.strip(".")
        self._client: nats.NATS | None = None
        self._connected = False

    @property
    def is_connected(self) -> bool:
        """Expose broker state for local diagnostics without leaking a broker URL."""
        return self._connected

    async def run(self) -> None:
        """Maintain a local subscription until application shutdown."""
        if not self._settings.nats_enabled:
            return
        while True:
            try:
                client = await nats.connect(
                    self._settings.nats_url,
                    reconnect_time_wait=self._settings.nats_reconnect_delay_seconds,
                    max_reconnect_attempts=-1,
                )
                self._client = client
                await client.subscribe(f"{self._subject_prefix}.>", cb=self._on_message)
                await client.flush()
                self._connected = True
                log.info("nats.connected", subject_prefix=self._subject_prefix)
                while client.is_connected:
                    await asyncio.sleep(1)
            except asyncio.CancelledError:
                raise
            except Exception as error:
                log.warning(
                    "nats.disconnected",
                    error_type=type(error).__name__,
                    retry_delay_seconds=self._settings.nats_reconnect_delay_seconds,
                )
                await asyncio.sleep(self._settings.nats_reconnect_delay_seconds)
            finally:
                self._connected = False
                self._client = None

    async def aclose(self) -> None:
        """Close the transport cleanly when the application stops."""
        client = self._client
        self._connected = False
        self._client = None
        if client is not None:
            await client.close()

    async def publish(self, event: AtlasEvent) -> bool:
        """Publish a normalized event once NATS is ready, without buffering unbounded work."""
        client = self._client
        if client is None or not self._connected:
            log.warning("nats.publish_skipped", event_type=event.type, reason="not_connected")
            return False
        try:
            await client.publish(self.subject_for(event.type), event.model_dump_json().encode())
        except asyncio.CancelledError:
            raise
        except Exception as error:
            log.warning("nats.publish_failed", event_type=event.type, error_type=type(error).__name__)
            return False
        return True

    def subject_for(self, event_type: str) -> str:
        """Map a dotted ATLAS type to one subject below the configured namespace."""
        return f"{self._subject_prefix}.{event_type}"

    def event_from_message(self, subject: str, payload: bytes) -> AtlasEvent | None:
        """Validate incoming NATS data and bind it to its declared subject."""
        prefix = f"{self._subject_prefix}."
        if not self._subject_prefix or not subject.startswith(prefix):
            log.warning("nats.message_rejected", subject=subject, reason="outside_event_namespace")
            return None
        event_type = subject.removeprefix(prefix)
        if not event_type:
            log.warning("nats.message_rejected", subject=subject, reason="missing_event_type")
            return None
        try:
            event = AtlasEvent.model_validate_json(payload)
        except ValueError:
            log.warning("nats.message_rejected", subject=subject, reason="invalid_event")
            return None
        if event.type != event_type:
            log.warning("nats.message_rejected", subject=subject, reason="event_type_mismatch")
            return None
        return event

    async def _on_message(self, message: object) -> None:
        """Persist valid bus events exactly once; SQLite de-duplicates their IDs."""
        subject = str(getattr(message, "subject", ""))
        payload = bytes(getattr(message, "data", b""))
        event = self.event_from_message(subject, payload)
        if event is not None:
            self._receive_event(event)
            log.info("nats.event_received", event_type=event.type, subject=subject)
