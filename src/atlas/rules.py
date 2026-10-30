"""Small deterministic reactions for safe, local home-event handling."""

from atlas.config import Settings
from atlas.events import AtlasEvent


class EventRuleEngine:
    """Translate selected normalized events into auditable, non-command reactions."""

    def __init__(self, settings: Settings) -> None:
        self._enabled = settings.event_rules_enabled

    def reactions_for(self, event: AtlasEvent) -> list[AtlasEvent]:
        """Return pure local reactions without calling an LLM or a device API."""
        if not self._enabled or event.type != "device.offline":
            return []
        return [
            AtlasEvent(
                type="alert.raised",
                source="rules",
                correlation_id=event.correlation_id,
                payload={
                    "severity": "warning",
                    "summary": f"{_device_label(event)} is offline.",
                    "source_event_id": str(event.id),
                },
            )
        ]


def _device_label(event: AtlasEvent) -> str:
    """Extract one bounded display label from an untrusted device payload."""
    data = event.payload.get("data")
    candidate: object | None = None
    if isinstance(data, dict):
        candidate = data.get("device") or data.get("name") or data.get("id")
    elif isinstance(data, str):
        candidate = data
    if candidate is None:
        return "A device"
    label = " ".join(str(candidate).split())
    return label[:100] if label else "A device"
