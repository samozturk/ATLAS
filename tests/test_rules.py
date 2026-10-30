from atlas.config import Settings
from atlas.events import AtlasEvent
from atlas.rules import EventRuleEngine


def test_device_offline_rule_creates_a_correlated_local_warning() -> None:
    event = AtlasEvent(
        type="device.offline",
        source="mqtt",
        payload={"data": {"device": "Hall sensor"}},
    )

    reactions = EventRuleEngine(Settings(event_rules_enabled=True)).reactions_for(event)

    assert len(reactions) == 1
    assert reactions[0].type == "alert.raised"
    assert reactions[0].source == "rules"
    assert reactions[0].correlation_id == event.correlation_id
    assert reactions[0].payload["summary"] == "Hall sensor is offline."


def test_event_rules_can_be_disabled_and_do_not_react_to_alerts() -> None:
    offline = AtlasEvent(type="device.offline", source="mqtt")
    alert = AtlasEvent(type="alert.raised", source="rules")

    assert EventRuleEngine(Settings(event_rules_enabled=False)).reactions_for(offline) == []
    assert EventRuleEngine(Settings()).reactions_for(alert) == []
