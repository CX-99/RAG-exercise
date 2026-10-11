import numpy as np
import pytest

from app.rag.chunking import Chunk
from app.rag.embeddings import normalize
from app.rag.retrieval import HybridRetriever, reciprocal_rank_fusion, tokenize
from tests.fakes import FakeEmbedder


def chunks() -> list[Chunk]:
    return [
        Chunk(chunk_id=str(i), doc_id=str(i), title="Policy", section="Rule", text=text)
        for i, text in enumerate(["holiday allowance", "quasar reimbursement", "remote devices"])
    ]


def test_rrf_combines_both_rankings_with_one_based_ranks() -> None:
    # Items 0 and 2 tie; the stable index breaks the tie ahead of item 1.
    assert reciprocal_rank_fusion([[0, 1, 2], [2, 1, 0]]) == [0, 2, 1]
    assert reciprocal_rank_fusion([[0, 1, 2], [1, 2, 0]]) == [1, 0, 2]
    assert reciprocal_rank_fusion([[2, 0, 1]]) == [2, 0, 1]


def test_dense_cosine_normalizes_vectors_and_indexes_heading_context() -> None:
    fake = FakeEmbedder([[2, 0], [0, 3], [-2, 0]], [8, 0])
    retriever = HybridRetriever(chunks(), fake)
    results = retriever.retrieve("unmatched", 2)
    assert [result.chunk.chunk_id for result in results] == ["0", "1"]
    assert [result.dense_score for result in results] == pytest.approx([1, 0])
    assert fake.indexed_texts[0] == "Policy — Rule\n\nholiday allowance"
    assert fake.queries == ["unmatched"]


def test_keyword_evidence_changes_ranking_without_changing_cosine_score() -> None:
    retriever = HybridRetriever(chunks(), FakeEmbedder([[1, 0], [0, 1], [-1, 0]], [1, 0]))
    results = retriever.retrieve("QUASAR", 3)
    assert [result.chunk.chunk_id for result in results] == ["1", "0", "2"]
    assert results[0].dense_score == pytest.approx(0)
    assert tokenize("Home-OFFICE, SGD 400!") == ["home", "office", "sgd", "400"]


def test_zero_vectors_stay_finite_and_ties_are_stable() -> None:
    assert np.array_equal(normalize(np.zeros((1, 2), dtype=np.float32)), [[0, 0]])
    retriever = HybridRetriever(chunks(), FakeEmbedder([[0, 0]] * 3, [0, 0]))
    assert [r.chunk.chunk_id for r in retriever.retrieve("unknown", 10)] == ["0", "1", "2"]


def test_invalid_index_and_query_shapes_are_rejected() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        HybridRetriever([], FakeEmbedder([], [1, 0]))
    with pytest.raises(ValueError, match="one non-empty"):
        HybridRetriever(chunks(), FakeEmbedder([[1, 0]], [1, 0]))
    retriever = HybridRetriever(chunks(), FakeEmbedder([[1, 0]] * 3, [1, 0, 0]))
    with pytest.raises(ValueError, match="dimensions"):
        retriever.retrieve("question", 5)
