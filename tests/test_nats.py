import asyncio

from atlas.config import Settings
from atlas.events import AtlasEvent
from atlas.nats import NATSEventBus


class FakeNATSClient:
    def __init__(self) -> None:
        self.published: list[tuple[str, bytes]] = []

    async def publish(self, subject: str, payload: bytes) -> None:
        self.published.append((subject, payload))


def test_nats_bus_publishes_a_normalized_event_when_connected() -> None:
    received: list[AtlasEvent] = []
    bus = NATSEventBus(
        Settings(nats_enabled=True, nats_event_subject_prefix="atlas.events"), received.append
    )
    client = FakeNATSClient()
    bus._client = client  # type: ignore[assignment]
    bus._connected = True
    event = AtlasEvent(type="motion.detected", source="mqtt", payload={"room": "hall"})

    assert asyncio.run(bus.publish(event)) is True
    assert client.published[0][0] == "atlas.events.motion.detected"
    assert AtlasEvent.model_validate_json(client.published[0][1]) == event


def test_nats_bus_accepts_only_matching_normalized_event_subjects() -> None:
    received: list[AtlasEvent] = []
    bus = NATSEventBus(Settings(nats_enabled=True), received.append)
    event = AtlasEvent(type="temperature.changed", source="sensor", payload={"celsius": 21.5})

    assert bus.event_from_message("atlas.events.temperature.changed", event.model_dump_json().encode()) == event
    assert bus.event_from_message("atlas.events.motion.detected", event.model_dump_json().encode()) is None
    assert bus.event_from_message("home.events.temperature.changed", event.model_dump_json().encode()) is None
