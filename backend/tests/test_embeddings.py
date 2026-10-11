import json
from collections.abc import Callable

import httpx2
import numpy as np
import pytest
from pydantic import SecretStr

from app.config import Settings
from app.rag.chunking import Chunk
from app.rag.embeddings import OpenRouterEmbedder
from app.rag.errors import LLMError
from app.rag.retrieval import HybridRetriever

Handler = Callable[[httpx2.Request], httpx2.Response]


def embedding_response(vectors: list[list[float]], order: list[int] | None = None) -> dict:
    order = order if order is not None else list(range(len(vectors)))
    return {
        "object": "list",
        "model": "openai/text-embedding-3-small",
        "data": [{"object": "embedding", "index": i, "embedding": vectors[i]} for i in order],
        "usage": {"prompt_tokens": 4, "total_tokens": 4},
    }


def make_embedder(settings: Settings, handler: Handler) -> OpenRouterEmbedder:
    settings.openrouter_base_url = "https://provider.invalid/api/v1"
    return OpenRouterEmbedder(
        settings, http_client=httpx2.Client(transport=httpx2.MockTransport(handler))
    )


def test_request_shape_and_rows_restored_to_input_order(settings: Settings) -> None:
    requests = []

    def respond(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        # Rows come back out of order; each carries its input's index.
        return httpx2.Response(200, json=embedding_response([[1, 0], [0, 1]], order=[1, 0]))

    embedder = make_embedder(settings, respond)
    vectors = embedder.embed_documents(["first", "second"])
    embedder.close()
    assert np.array_equal(vectors, [[1, 0], [0, 1]])
    request = requests[0]
    assert str(request.url) == "https://provider.invalid/api/v1/embeddings"
    assert request.headers["authorization"] == "Bearer fake-test-key"
    body = json.loads(request.content)
    assert body["model"] == "openai/text-embedding-3-small"
    assert body["input"] == ["first", "second"]


def test_a_query_is_one_row(settings: Settings) -> None:
    embedder = make_embedder(
        settings, lambda request: httpx2.Response(200, json=embedding_response([[0.6, 0.8]]))
    )
    assert embedder.embed_query("question").shape == (1, 2)


def test_a_row_count_mismatch_is_a_safe_error(settings: Settings) -> None:
    embedder = make_embedder(
        settings, lambda request: httpx2.Response(200, json=embedding_response([[1, 0]]))
    )
    with pytest.raises(LLMError) as caught:
        embedder.embed_documents(["first", "second"])
    assert caught.value.kind == "EmbeddingCountMismatch"


@pytest.mark.parametrize("failure", ["status", "timeout"])
def test_provider_failures_become_safe_errors(settings: Settings, failure: str) -> None:
    def respond(request: httpx2.Request) -> httpx2.Response:
        if failure == "timeout":
            raise httpx2.ReadTimeout("private question must not escape", request=request)
        return httpx2.Response(401, json={"error": {"message": "private question must not escape"}})

    with pytest.raises(LLMError, match="temporarily unavailable") as caught:
        make_embedder(settings, respond).embed_query("private question")
    assert "private question" not in str(caught.value)
    if failure == "status":
        assert caught.value.status_code == 401
    else:
        assert caught.value.kind == "APITimeoutError"


@pytest.mark.parametrize("key", [None, SecretStr(""), SecretStr("  ")])
def test_a_missing_key_fails_fast(settings: Settings, key: SecretStr | None) -> None:
    settings.openrouter_api_key = key
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        OpenRouterEmbedder(settings)


def test_the_retriever_ranks_with_hosted_embeddings(settings: Settings) -> None:
    chunks = [
        Chunk(chunk_id=str(i), doc_id=str(i), title="Policy", section="Rule", text=text)
        for i, text in enumerate(["holiday allowance", "remote devices"])
    ]

    def respond(request: httpx2.Request) -> httpx2.Response:
        texts = json.loads(request.content)["input"]
        # Documents get orthogonal vectors; the query points at the second document.
        vectors = [[1.0, 0.0], [0.0, 1.0]] if len(texts) == 2 else [[0.1, 0.9]]
        return httpx2.Response(200, json=embedding_response(vectors))

    retriever = HybridRetriever(chunks, make_embedder(settings, respond))
    results = retriever.retrieve("laptops at home", 2)
    assert [result.chunk.chunk_id for result in results] == ["1", "0"]
    assert results[0].dense_score == pytest.approx(0.9939, abs=1e-4)
