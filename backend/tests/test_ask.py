import re

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.rag.pipeline import get_pipeline
from app.schemas import MAX_QUESTION_CHARS
from tests.fakes import FakePipeline


def test_ask_returns_answer_citations_and_request_id(client: TestClient) -> None:
    response = client.post("/api/ask", json={"question": "How many days of annual leave?"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"]
    assert isinstance(body["citations"], list)
    assert re.fullmatch(r"[0-9a-f]{32}", body["request_id"])


def test_each_request_gets_a_distinct_request_id(client: TestClient) -> None:
    first = client.post("/api/ask", json={"question": "a"}).json()["request_id"]
    second = client.post("/api/ask", json={"question": "a"}).json()["request_id"]

    assert first != second


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"question": ""},
        {"question": "   "},
        {"question": "x" * (MAX_QUESTION_CHARS + 1)},
    ],
    ids=["missing", "empty", "whitespace", "too-long"],
)
def test_ask_rejects_invalid_questions(client: TestClient, payload: dict) -> None:
    response = client.post("/api/ask", json=payload)

    assert response.status_code == 422


def test_ask_delegates_to_injected_pipeline(client: TestClient) -> None:
    fake = FakePipeline()
    app.dependency_overrides[get_pipeline] = lambda: fake

    response = client.post("/api/ask", json={"question": "  How much leave?  "})

    assert response.status_code == 200
    assert fake.questions == ["How much leave?"]
    body = response.json()
    assert body["answer"] == "Employees get 14 days."
    assert body["citations"] == [
        {
            "doc_id": "hr-1",
            "title": "Leave Policy",
            "section": "3. Leave",
            "snippet": "14 days",
            "score": 0.9,
        }
    ]
