import asyncio

import httpx
from fastapi.testclient import TestClient

from atlas.config import Settings
from atlas.llm.ollama import OllamaModelCatalog
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


def test_system_health_reports_configured_ollama_models() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/tags"
        return httpx.Response(
            200,
            json={
                "models": [
                    {"name": "qwen3.6:35b"},
                    {"name": "qwen3.5:122b-a10b"},
                    {"name": "embeddinggemma:latest"},
                ]
            },
        )

    settings = Settings(environment="test")
    client = httpx.AsyncClient(
        base_url="http://ollama.test", transport=httpx.MockTransport(handler)
    )
    catalog = OllamaModelCatalog(settings, client=client)
    app = create_app(settings, ollama_catalog=catalog)

    with TestClient(app) as test_client:
        response = test_client.get("/system-health")

    asyncio.run(client.aclose())
    assert response.status_code == 200
    assert response.json()["status"] == "ready"
    assert response.json()["ollama_reachable"] is True
    assert [model["installed"] for model in response.json()["models"]] == [True, True, True]
