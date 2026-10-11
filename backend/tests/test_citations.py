import pytest

from app.rag.citations import process_citations
from app.rag.prompts import NOT_FOUND
from app.rag.retrieval import RetrievedChunk


def test_first_appearance_order_repeated_and_invalid_markers(sources: list[RetrievedChunk]) -> None:
    result = process_citations("Third [3], first [1], third again [3]. Bad [99][0].", sources)
    assert result.answer == "Third [1], first [2], third again [1]. Bad ."
    assert [citation.section for citation in result.citations] == [
        sources[2].chunk.section,
        sources[0].chunk.section,
    ]
    assert result.citations[0].snippet == sources[2].chunk.text
    assert result.citations[0].score == sources[2].dense_score


@pytest.mark.parametrize("suffix", ["", " [2]", " [999]", " [1, 2–3]"])
def test_not_found_has_no_citations(sources: list[RetrievedChunk], suffix: str) -> None:
    result = process_citations(NOT_FOUND + suffix, sources)
    assert result.answer == NOT_FOUND
    assert result.citations == []


def test_uncited_and_non_numeric_brackets_are_preserved(sources: list[RetrievedChunk]) -> None:
    uncited = process_citations("Text with [a label].", sources)
    assert uncited.answer == NOT_FOUND
    assert uncited.citations == []
    cited = process_citations("Text with [a label] and evidence [2].", sources)
    assert cited.answer == "Text with [a label] and evidence [1]."
    assert len(cited.citations) == 1


def test_grouped_markers_ranges_and_first_appearance(sources: list[RetrievedChunk]) -> None:
    result = process_citations("First [3, 1-2, 99]. Later [2–3]. Invalid [0, 9–12].", sources)
    assert result.answer == "First [1, 2, 3]. Later [3, 1]. Invalid ."
    assert [citation.section for citation in result.citations] == [
        sources[2].chunk.section,
        sources[0].chunk.section,
        sources[1].chunk.section,
    ]
    assert process_citations("Fact one [1, 2]. Fact two [3].", sources).answer == (
        "Fact one [1, 2]. Fact two [3]."
    )


@pytest.mark.parametrize("text", ["Employees get 30 days.", "Unsupported [99].", "Text [0, 9-12]."])
def test_no_valid_citation_means_refusal(sources: list[RetrievedChunk], text: str) -> None:
    result = process_citations(text, sources)
    assert result.answer == NOT_FOUND
    assert result.citations == []


def test_ranges_are_bounded_and_duplicate_references_collapsed(
    sources: list[RetrievedChunk],
) -> None:
    huge = "9" * 5000
    result = process_citations(f"Fact [0–{huge}, 2, 3]. More [3-1].", sources)
    assert result.answer == "Fact [1, 2, 3]. More [3, 2, 1]."
    assert len(result.citations) == 3


@pytest.mark.parametrize("marker", ["【1】", "［1］"])
def test_lenticular_and_fullwidth_brackets_are_citations(
    sources: list[RetrievedChunk], marker: str
) -> None:
    result = process_citations(f"The nightly hotel limit is SGD 220{marker}.", sources)
    assert result.answer == "The nightly hotel limit is SGD 220[1]."
    assert [citation.section for citation in result.citations] == [sources[0].chunk.section]


def test_not_found_with_lenticular_marker_is_still_not_found(
    sources: list[RetrievedChunk],
) -> None:
    assert process_citations(NOT_FOUND + "【2】", sources).answer == NOT_FOUND
