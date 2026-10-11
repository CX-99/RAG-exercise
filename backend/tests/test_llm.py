import asyncio
import json

import httpx2
import pytest
from pydantic import SecretStr

from app.config import Settings
from app.rag.llm import LLMError, OpenRouterLLM, TokenUsage, TruncatedCompletion
from app.rag.prompts import (
    CONVERSATION_PROMPT,
    GENERAL_KNOWLEDGE_PREFIX,
    NOT_FOUND,
    SYSTEM_PROMPT,
)
from app.rag.retrieval import RetrievedChunk


@pytest.mark.parametrize("key", [None, SecretStr(""), SecretStr("  ")])
def test_openrouter_fails_fast_without_key(settings: Settings, key: SecretStr | None) -> None:
    settings.openrouter_api_key = key
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        OpenRouterLLM(settings)


def test_openrouter_request_shape_and_usage(
    settings: Settings, sources: list[RetrievedChunk]
) -> None:
    requests = []
    settings.openrouter_api_key = SecretStr("fake-test-key")
    settings.openrouter_base_url = "https://provider.invalid/api/v1"
    settings.llm_model = "test/model"
    settings.llm_max_tokens = 123
    settings.llm_timeout_seconds = 7

    def respond(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return httpx2.Response(
            200,
            json={
                "id": "fake",
                "object": "chat.completion",
                "created": 0,
                "model": "test/model",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Fact [1]."},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 40, "completion_tokens": 4, "total_tokens": 44},
            },
        )

    async def exercise() -> None:
        async with httpx2.AsyncClient(transport=httpx2.MockTransport(respond)) as http:
            llm = OpenRouterLLM(settings, http_client=http)
            try:
                assert llm._client.max_retries == 2
                reply = await llm.generate("What is the rule?", sources)
                assert reply.text == "Fact [1]."
                assert reply.usage is not None and reply.usage.total_tokens == 44
            finally:
                await llm.aclose()

    asyncio.run(exercise())
    assert len(requests) == 1
    request = requests[0]
    assert str(request.url) == "https://provider.invalid/api/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer fake-test-key"
    body = json.loads(request.content)
    assert (body["model"], body["temperature"], body["max_tokens"]) == ("test/model", 0, 123)
    assert body["messages"][0] == {"role": "system", "content": SYSTEM_PROMPT}
    payload = json.loads(body["messages"][1]["content"])
    assert payload["question"] == "What is the rule?"
    assert payload["sources"][0] == {
        "number": 1,
        "title": "Leave Policy",
        "section": sources[0].chunk.section,
        "text": sources[0].chunk.text,
    }
    assert request.extensions["timeout"]["read"] == 7


@pytest.mark.parametrize("failure", ["status", "timeout", "empty"])
def test_openrouter_maps_errors_and_retries_only_mock_http(
    settings: Settings, sources: list[RetrievedChunk], failure: str
) -> None:
    settings.openrouter_api_key = SecretStr("fake-test-key")
    requests = []

    def respond(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        if failure == "timeout":
            raise httpx2.ReadTimeout("private question must not escape", request=request)
        if failure == "status":
            return httpx2.Response(
                503,
                headers={"retry-after-ms": "1"},
                json={"error": {"message": "private question must not escape"}},
            )
        return httpx2.Response(
            200,
            json={
                "id": "fake",
                "object": "chat.completion",
                "created": 0,
                "model": "fake",
                "choices": [],
            },
        )

    async def exercise() -> None:
        async with httpx2.AsyncClient(transport=httpx2.MockTransport(respond)) as http:
            llm = OpenRouterLLM(settings, http_client=http)
            try:
                with pytest.raises(LLMError, match="temporarily unavailable") as caught:
                    await llm.generate("Question?", sources)
                assert "private question" not in str(caught.value)
                if failure == "status":
                    assert caught.value.status_code == 503
                if failure == "timeout":
                    assert caught.value.kind == "APITimeoutError"
            finally:
                await llm.aclose()

    asyncio.run(exercise())
    assert len(requests) == (1 if failure == "empty" else 3)


def test_openrouter_truncated_completion_keeps_its_token_usage(
    settings: Settings, sources: list[RetrievedChunk]
) -> None:
    settings.openrouter_api_key = SecretStr("fake-test-key")

    def respond(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(
            200,
            json={
                "id": "fake",
                "object": "chat.completion",
                "created": 0,
                "model": "fake",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "length",
                        "message": {"role": "assistant", "content": "Unfinished [1]"},
                    }
                ],
                "usage": {"prompt_tokens": 900, "completion_tokens": 50, "total_tokens": 950},
            },
        )

    async def exercise() -> None:
        async with httpx2.AsyncClient(transport=httpx2.MockTransport(respond)) as http:
            llm = OpenRouterLLM(settings, http_client=http)
            try:
                with pytest.raises(TruncatedCompletion) as caught:
                    await llm.generate("Question?", sources)
                assert caught.value.kind == "TruncatedCompletion"
                assert caught.value.usage == TokenUsage(
                    prompt_tokens=900, completion_tokens=50, total_tokens=950
                )
            finally:
                await llm.aclose()

    asyncio.run(exercise())


@pytest.mark.parametrize("missing", ["prompt_tokens", "completion_tokens", "total_tokens"])
@pytest.mark.parametrize("null", [False, True])
def test_openrouter_incomplete_usage_does_not_discard_answer(
    settings: Settings, sources: list[RetrievedChunk], missing: str, null: bool
) -> None:
    settings.openrouter_api_key = SecretStr("fake-test-key")
    usage: dict[str, int | None] = {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}
    if null:
        usage[missing] = None
    else:
        del usage[missing]

    def respond(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(
            200,
            json={
                "id": "fake",
                "object": "chat.completion",
                "created": 0,
                "model": "fake",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "Supported [1]."},
                    }
                ],
                "usage": usage,
            },
        )

    async def exercise() -> None:
        async with httpx2.AsyncClient(transport=httpx2.MockTransport(respond)) as http:
            llm = OpenRouterLLM(settings, http_client=http)
            try:
                reply = await llm.generate("Question?", sources)
                assert reply.text == "Supported [1]."
                assert reply.usage is None
            finally:
                await llm.aclose()

    asyncio.run(exercise())


def test_openrouter_converse_sends_the_message_without_sources(settings: Settings) -> None:
    settings.openrouter_api_key = SecretStr("fake-test-key")
    requests = []

    def respond(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return httpx2.Response(
            200,
            json={
                "id": "fake",
                "object": "chat.completion",
                "created": 0,
                "model": "fake",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "Hello! Ask me about leave."},
                    }
                ],
            },
        )

    async def exercise() -> None:
        async with httpx2.AsyncClient(transport=httpx2.MockTransport(respond)) as http:
            llm = OpenRouterLLM(settings, http_client=http)
            try:
                reply = await llm.converse("hi")
                assert reply.text == "Hello! Ask me about leave."
            finally:
                await llm.aclose()

    asyncio.run(exercise())
    body = json.loads(requests[0].content)
    assert body["messages"] == [
        {"role": "system", "content": CONVERSATION_PROMPT},
        {"role": "user", "content": "hi"},
    ]
    assert body["temperature"] == 0


def test_conversation_prompt_labels_general_knowledge_and_forbids_company_guesses() -> None:
    assert GENERAL_KNOWLEDGE_PREFIX in CONVERSATION_PROMPT
    assert CONVERSATION_PROMPT.endswith(NOT_FOUND)
    assert "never guess" in CONVERSATION_PROMPT
