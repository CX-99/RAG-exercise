from fastapi.testclient import TestClient

from app import __version__


def test_health_reports_ok_and_version(client: TestClient) -> None:
    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "version": __version__}
