import os
import socket
from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.config import Settings, get_settings
from app.rag.chunking import Chunk
from app.rag.retrieval import RetrievedChunk
from tests.fakes import FakePipeline, make_settings

# A deliberately invalid key: tests block outside sockets, so it can never reach the API.
FAKE_KEY = "fake-test-key"


def pytest_configure() -> None:
    # Runs before test module imports create the app: never read a developer's .env, and
    # give the app the key it requires at startup.
    Settings.model_config["env_file"] = None
    os.environ["OPENROUTER_API_KEY"] = FAKE_KEY
    get_settings.cache_clear()


@pytest.fixture(autouse=True)
def offline(monkeypatch: pytest.MonkeyPatch) -> None:
    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex
    original_getaddrinfo = socket.getaddrinfo

    def loopback(address: Any) -> bool:
        return isinstance(address, tuple) and address[0] in {"127.0.0.1", "::1"}

    def connect(sock: socket.socket, address: Any) -> None:
        # Windows asyncio uses a loopback socket pair for its internal wake-up pipe.
        if not loopback(address):
            raise AssertionError("Tests must not access the network")
        original_connect(sock, address)

    def connect_ex(sock: socket.socket, address: Any) -> int:
        if not loopback(address):
            raise AssertionError("Tests must not access the network")
        return original_connect_ex(sock, address)

    def getaddrinfo(host: str, *args: Any, **kwargs: Any) -> list:
        if host not in {"127.0.0.1", "::1", "localhost"}:
            raise AssertionError("Tests must not resolve external hosts")
        return original_getaddrinfo(host, *args, **kwargs)

    monkeypatch.setattr(socket.socket, "connect", connect)
    monkeypatch.setattr(socket.socket, "connect_ex", connect_ex)
    monkeypatch.setattr(socket, "getaddrinfo", getaddrinfo)


@pytest.fixture
def settings() -> Settings:
    return make_settings(openrouter_api_key=FAKE_KEY)


@pytest.fixture
def sources() -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            chunk=Chunk(
                chunk_id=f"leave:{i}",
                doc_id="leave",
                title="Leave Policy",
                section=f"3. Leave > 3.{i + 1} Rule",
                text=f"Rule {i + 1} applies. Keep records.",
            ),
            dense_score=0.9 - i * 0.1,
        )
        for i in range(3)
    ]


@pytest.fixture
def client() -> Iterator[TestClient]:
    from app.main import app
    from app.rag.pipeline import get_pipeline

    app.dependency_overrides[get_pipeline] = lambda: FakePipeline()
    try:
        with TestClient(app) as test_client:
            yield test_client
    finally:
        app.dependency_overrides.clear()
