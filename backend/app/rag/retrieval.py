"""In-memory dense + BM25 search with deterministic reciprocal rank fusion."""

import re
from collections.abc import Sequence
from typing import Protocol

import numpy as np
from pydantic import BaseModel
from rank_bm25 import BM25Okapi

from app.rag.chunking import Chunk
from app.rag.embeddings import Embedder, normalize


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens. Punctuation splits words, so "What's" gives "what" and "s"."""
    return re.findall(r"\b\w+\b", text.lower())


def reciprocal_rank_fusion(rankings: Sequence[Sequence[int]], k: int = 60) -> list[int]:
    """Fuse ranked lists of item indices with Reciprocal Rank Fusion (Cormack et al., 2009).

    Each item scores ``sum(1 / (k + rank))`` over the lists it appears in, with ranks
    starting at 1. RRF uses only ranks, so dense and BM25 scores never need a common
    scale. Ties go to the lower index, which keeps results deterministic.

    Args:
        rankings: Lists of item indices, best first. A list may omit items.
        k: Smoothing constant; 60 is the value recommended in the original paper.

    Returns:
        Every item that appears in any list, best first.

    Raises:
        ValueError: If ``k`` is less than 1.
    """
    if k < 1:
        raise ValueError("RRF k must be positive")
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, index in enumerate(ranking, start=1):
            scores[index] = scores.get(index, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=lambda index: (-scores[index], index))


class RetrievedChunk(BaseModel):
    """A retrieved chunk and its dense cosine score, shown with each citation."""

    chunk: Chunk
    dense_score: float


class Retriever(Protocol):
    """Ranks corpus chunks for a question; implemented by HybridRetriever and test fakes."""

    def retrieve(self, question: str, top_k: int) -> list[RetrievedChunk]:
        """Return up to ``top_k`` chunks, best first."""
        ...


class HybridRetriever:
    """Dense plus BM25 search over an in-memory corpus, fused with RRF.

    The corpus is embedded once, at construction. Dense search is a brute-force matrix
    product: exact, and fast enough for a small corpus.

    Args:
        chunks: The corpus chunks to index.
        embedder: Embeds both the chunks and each query.

    Raises:
        ValueError: If ``chunks`` is empty or the embedder returns malformed vectors.
    """

    def __init__(self, chunks: Sequence[Chunk], embedder: Embedder) -> None:
        if not chunks:
            raise ValueError("Corpus contains no non-empty chunks")
        self.chunks = list(chunks)
        self.embedder = embedder
        texts = [chunk.indexed_text for chunk in chunks]
        self.vectors = normalize(embedder.embed_documents(texts))
        if self.vectors.shape[0] != len(chunks) or self.vectors.shape[1] == 0:
            raise ValueError("Expected one non-empty embedding per chunk")
        self.bm25 = BM25Okapi([tokenize(text) for text in texts])

    def retrieve(self, question: str, top_k: int) -> list[RetrievedChunk]:
        """Return the ``top_k`` best chunks for ``question`` by fused rank.

        Each result keeps its dense cosine score even though ranking uses RRF: an absolute
        similarity is more informative to show with a citation than a fused rank.

        Raises:
            ValueError: If ``top_k`` is less than 1, or the query embedding's dimension
                differs from the corpus embeddings'.
        """
        if top_k < 1:
            raise ValueError("top_k must be positive")
        query = normalize(self.embedder.embed_query(question))
        if query.shape != (1, self.vectors.shape[1]):
            raise ValueError("Query and document embedding dimensions differ")
        dense_scores = np.clip(self.vectors @ query[0], -1.0, 1.0)
        sparse_scores = self.bm25.get_scores(tokenize(question))
        dense_order = np.argsort(-dense_scores, kind="stable").tolist()
        # Zero/negative keyword matches carry no evidence; do not promote arbitrary ties.
        sparse_order = [
            int(index)
            for index in np.argsort(-sparse_scores, kind="stable")
            if sparse_scores[index] > 0
        ]
        fused = reciprocal_rank_fusion([dense_order, sparse_order])
        return [
            RetrievedChunk(chunk=self.chunks[i], dense_score=float(dense_scores[i]))
            for i in fused[:top_k]
        ]
