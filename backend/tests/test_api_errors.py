import logging

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.rag.llm import LLMError, TokenUsage
from app.rag.pipeline import get_pipeline
from app.rag.telemetry import request_metrics
from app.schemas import RagAnswer


@pytest.mark.parametrize(
    "error",
    [
        LLMError("RateLimitError", 429),
        LLMError("EmptyCompletion"),
        TimeoutError("private details"),
    ],
)
def test_generation_failure_is_503_with_safe_log(
    client: TestClient, caplog: pytest.LogCaptureFixture, error: Exception
) -> None:
    class FailingPipeline:
        async def answer(self, question: str) -> RagAnswer:
            raise error

    app.dependency_overrides[get_pipeline] = lambda: FailingPipeline()
    with caplog.at_level(logging.INFO, logger="app.api.ask"):
        response = client.post("/api/ask", json={"question": "Secret personal details?"})
    assert response.status_code == 503
    assert response.json() == {"detail": "Answer generation is temporarily unavailable."}
    logs = [r.getMessage() for r in caplog.records if r.name == "app.api.ask"]
    assert len(logs) == 1
    assert "request_id=" in logs[0]
    assert "retrieval_ms=" in logs[0] and "generation_ms=" in logs[0]
    assert "chunk_count=" in logs[0] and "refused=no" in logs[0]
    assert "error_type=" in logs[0]
    assert "Secret personal details" not in caplog.text
    assert "private details" not in caplog.text


def test_one_request_log_contains_usage_without_question(
    client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    class MeasuredPipeline:
        async def answer(self, question: str) -> RagAnswer:
            metrics = request_metrics.get()
            assert metrics is not None
            metrics.retrieval_ms = 12
            metrics.generation_ms = 5
            metrics.chunk_count = 3
            metrics.usage = TokenUsage(prompt_tokens=20, completion_tokens=5, total_tokens=25)
            return RagAnswer(answer="An answer.")

    app.dependency_overrides[get_pipeline] = lambda: MeasuredPipeline()
    with caplog.at_level(logging.INFO, logger="app.api.ask"):
        response = client.post("/api/ask", json={"question": "Private question?"})
    logs = [r.getMessage() for r in caplog.records if r.name == "app.api.ask"]
    assert response.status_code == 200
    assert len(logs) == 1 and response.json()["request_id"] in logs[0]
    assert "retrieval_ms=12.00" in logs[0]
    assert "'total_tokens': 25" in logs[0]
    assert "Private question" not in caplog.text
    assert request_metrics.get() is None


def test_unexpected_error_is_500_and_logged_at_error_without_details(
    client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    class BrokenPipeline:
        async def answer(self, question: str) -> RagAnswer:
            raise RuntimeError("private question or provider body")

    app.dependency_overrides[get_pipeline] = lambda: BrokenPipeline()
    with (
        caplog.at_level(logging.INFO, logger="app.api.ask"),
        TestClient(app, raise_server_exceptions=False) as non_raising_client,
    ):
        response = non_raising_client.post("/api/ask", json={"question": "Secret question?"})
    assert response.status_code == 500
    records = [record for record in caplog.records if record.name == "app.api.ask"]
    assert len(records) == 1 and records[0].levelno == logging.ERROR
    assert "error_type=RuntimeError" in records[0].getMessage()
    assert "error_type=none" not in records[0].getMessage()
    assert "private question" not in caplog.text
    assert "Secret question" not in caplog.text
    assert request_metrics.get() is None


def test_construction_failure_is_generic_503_and_logs_type_only(
    client: TestClient, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    import asyncio

    from app.rag import pipeline

    def fail(settings: object) -> None:
        raise OSError("secret key and local path")

    app.dependency_overrides.pop(get_pipeline)
    monkeypatch.setattr(pipeline, "_pipeline", None)
    monkeypatch.setattr(pipeline, "_build_lock", asyncio.Lock())
    monkeypatch.setattr(pipeline, "build_pipeline", fail)
    with caplog.at_level(logging.ERROR, logger="app.main"):
        response = client.post("/api/ask", json={"question": "Secret question?"})
    assert response.status_code == 503
    assert response.json() == {"detail": "Policy assistant is temporarily unavailable."}
    records = [record for record in caplog.records if record.name == "app.main"]
    assert len(records) == 1
    assert records[0].getMessage() == "pipeline_construction error_type=OSError"
    assert "secret key" not in caplog.text
    assert "Secret question" not in caplog.text


def test_refusal_reason_is_in_request_log(
    client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    from app.rag.prompts import NOT_FOUND

    class RefusingPipeline:
        async def answer(self, question: str) -> RagAnswer:
            metrics = request_metrics.get()
            assert metrics is not None
            metrics.refused = True
            metrics.refusal_reason = "uncited"
            return RagAnswer(answer=NOT_FOUND)

    app.dependency_overrides[get_pipeline] = lambda: RefusingPipeline()
    with caplog.at_level(logging.INFO, logger="app.api.ask"):
        response = client.post("/api/ask", json={"question": "Question?"})
    assert response.status_code == 200
    assert "refused=yes refusal_reason=uncited" in caplog.text
