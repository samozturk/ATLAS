from atlas.events import AtlasEvent


def test_event_generates_trace_metadata() -> None:
    event = AtlasEvent(type="sensor.motion")

    assert event.source == "atlas"
    assert event.occurred_at.tzinfo is not None
    assert event.id != event.correlation_id
