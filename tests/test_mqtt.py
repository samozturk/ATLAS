from fastapi.testclient import TestClient

from atlas.config import Settings
from atlas.conversations import ConversationStore
from atlas.main import create_app
from atlas.mqtt import MQTTEventAdapter


def test_mqtt_adapter_accepts_only_the_event_namespace_and_decodes_json(tmp_path) -> None:
    store = ConversationStore(str(tmp_path / "atlas.db"))
    store.initialize()
    adapter = MQTTEventAdapter(
        Settings(
            environment="test",
            mqtt_enabled=True,
            mqtt_host="mqtt.local",
            mqtt_event_topic_prefix="atlas/events",
        ),
        store.record_event,
    )

    event = adapter.event_from_message(
        "atlas/events/temperature.changed", b'{"device":"hall","celsius":21.5}'
    )

    assert event is not None
    assert event.type == "temperature.changed"
    assert event.source == "mqtt"
    assert event.payload == {
        "topic": "atlas/events/temperature.changed",
        "data": {"device": "hall", "celsius": 21.5},
    }
    store.record_event(event)
    assert store.list_events()[0].type == "temperature.changed"
    assert adapter.event_from_message("home/hall/temperature", b"21.5") is None


def test_events_endpoint_returns_persisted_local_events(tmp_path) -> None:
    app = create_app(Settings(environment="test", database_path=str(tmp_path / "atlas.db")))
    with TestClient(app) as client:
        event = app.state.mqtt_adapter.event_from_message(
            "atlas/events/motion.detected", b"hallway"
        )
        assert event is not None
        app.state.conversation_store.record_event(event)
        response = client.get("/events")

    assert response.status_code == 200
    assert response.json()[0]["type"] == "motion.detected"
    assert response.json()[0]["payload"] == {
        "topic": "atlas/events/motion.detected",
        "data": "hallway",
    }
