from fastapi.testclient import TestClient

from atlas.config import Settings
from atlas.main import create_app


def test_health_checks_report_running_service() -> None:
    app = create_app(Settings(app_name="ATLAS Test", environment="test"))
    with TestClient(app) as client:
        live = client.get("/health/live")
        ready = client.get("/health/ready")

    assert live.status_code == 200
    assert live.json() == {"status": "ok", "service": "ATLAS Test", "environment": "test"}
    assert ready.status_code == 200
    assert ready.json()["status"] == "ok"
