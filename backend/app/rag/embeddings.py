"""Dense embeddings: the seam the retriever uses, and the hosted implementation."""

from collections.abc import Sequence
from typing import Protocol

import httpx2
import numpy as np
from numpy.typing import NDArray
from openai import APIError, OpenAI

from app.config import Settings
from app.rag.errors import LLMError

VectorMatrix = NDArray[np.float32]


class Embedder(Protocol):
    """Turns text into dense vectors; implemented by OpenRouterEmbedder and test fakes."""

    def embed_documents(self, texts: Sequence[str]) -> VectorMatrix:
        """Return one row per text."""
        ...

    def embed_query(self, text: str) -> VectorMatrix:
        """Return a one-row matrix for a query."""
        ...


def normalize(vectors: VectorMatrix) -> VectorMatrix:
    """Scale each row to unit length, so a dot product equals cosine similarity.

    All-zero rows stay zero instead of becoming NaN.

    Raises:
        ValueError: If the input is not a finite two-dimensional matrix.
    """
    array = np.asarray(vectors, dtype=np.float32)
    if array.ndim != 2 or not np.isfinite(array).all():
        raise ValueError("Embeddings must be a finite two-dimensional matrix")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    return np.divide(array, norms, out=np.zeros_like(array), where=norms != 0)


class OpenRouterEmbedder:
    """Hosted embeddings through OpenRouter's OpenAI-compatible embeddings API.

    The client is synchronous because the retriever runs in a worker thread. Documents and
    queries use the same model with no instruction prefix, which suits OpenAI's embedding
    models.

    Args:
        settings: Supplies the key, base URL, embedding model and timeout.
        http_client: Optional HTTP client; tests inject one with a mock transport.

    Raises:
        ValueError: If the API key is missing or blank.
    """

    def __init__(self, settings: Settings, http_client: httpx2.Client | None = None) -> None:
        key = settings.openrouter_api_key
        if key is None or not key.get_secret_value().strip():
            raise ValueError("OPENROUTER_API_KEY is required for embeddings")
        self._model = settings.embedding_model
        self._client = OpenAI(
            api_key=key.get_secret_value(),
            base_url=settings.openrouter_base_url,
            timeout=settings.llm_timeout_seconds,
            max_retries=2,
            http_client=http_client,
        )

    def embed_documents(self, texts: Sequence[str]) -> VectorMatrix:
        """Embed document texts in one request, one row per text."""
        return self._embed(list(texts))

    def embed_query(self, text: str) -> VectorMatrix:
        """Embed one query."""
        return self._embed([text])

    def _embed(self, texts: list[str]) -> VectorMatrix:
        """Call the embeddings API and map every failure to :class:`LLMError`."""
        try:
            response = self._client.embeddings.create(model=self._model, input=texts)
        except (APIError, httpx2.HTTPError, TimeoutError) as error:
            raise LLMError(type(error).__name__, getattr(error, "status_code", None)) from error
        # The API may return rows out of order; each carries its input's index.
        rows = sorted(response.data or [], key=lambda item: item.index)
        if len(rows) != len(texts):
            raise LLMError("EmbeddingCountMismatch")
        return np.asarray([row.embedding for row in rows], dtype=np.float32)

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._client.close()
