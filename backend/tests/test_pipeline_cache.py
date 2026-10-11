import asyncio
import threading
import time

import pytest

from app.rag import pipeline
from tests.fakes import FakePipeline


def test_concurrent_first_requests_build_once_and_cached_calls_skip_lock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []
    fake = FakePipeline()

    def build(settings: object) -> FakePipeline:
        calls.append(threading.get_ident())
        time.sleep(0.01)
        return fake

    class ForbiddenLock:
        async def __aenter__(self) -> None:
            raise AssertionError("An initialized pipeline must bypass the lock")

        async def __aexit__(self, *args: object) -> None:
            pass

    async def exercise() -> None:
        results = await asyncio.gather(*(pipeline.get_pipeline() for _ in range(5)))
        assert all(result is fake for result in results)
        monkeypatch.setattr(pipeline, "_build_lock", ForbiddenLock())
        assert await pipeline.get_pipeline() is fake

    monkeypatch.setattr(pipeline, "build_pipeline", build)
    monkeypatch.setattr(pipeline, "_build_lock", asyncio.Lock())
    monkeypatch.setattr(pipeline, "_pipeline", None)
    asyncio.run(exercise())
    assert len(calls) == 1
    assert calls[0] != threading.get_ident()


def test_failed_construction_is_not_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    fake = FakePipeline()

    def build(settings: object) -> FakePipeline:
        calls.append(settings)
        if len(calls) == 1:
            raise OSError("private model path or key")
        return fake

    async def exercise() -> None:
        with pytest.raises(pipeline.PipelineUnavailable) as caught:
            await pipeline.get_pipeline()
        assert caught.value.kind == "OSError"
        assert "private" not in str(caught.value)
        assert await pipeline.get_pipeline() is fake

    monkeypatch.setattr(pipeline, "build_pipeline", build)
    monkeypatch.setattr(pipeline, "_build_lock", asyncio.Lock())
    monkeypatch.setattr(pipeline, "_pipeline", None)
    asyncio.run(exercise())
    assert len(calls) == 2


def test_requests_waiting_on_a_failed_build_fail_with_it(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    fake = FakePipeline()

    def build(settings: object) -> FakePipeline:
        calls.append(settings)
        if len(calls) == 1:
            time.sleep(0.05)  # a slow provider, so the other requests queue up
            raise TimeoutError("private details")
        return fake

    async def exercise() -> None:
        results = await asyncio.gather(
            *(pipeline.get_pipeline() for _ in range(3)), return_exceptions=True
        )
        failures = [r for r in results if isinstance(r, pipeline.PipelineUnavailable)]
        assert len(failures) == 3
        assert {failure.kind for failure in failures} == {"TimeoutError"}
        # Only one slow build ran; a request after the failure tries again.
        assert len(calls) == 1
        assert await pipeline.get_pipeline() is fake

    monkeypatch.setattr(pipeline, "build_pipeline", build)
    monkeypatch.setattr(pipeline, "_build_lock", asyncio.Lock())
    monkeypatch.setattr(pipeline, "_pipeline", None)
    asyncio.run(exercise())
    assert len(calls) == 2
