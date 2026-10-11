import asyncio

import pytest

from app.config import Settings
from app.rag.chunking import chunk_document
from app.rag.documents import load_documents
from app.rag.llm import LLMError, LLMReply
from app.rag.prompts import NOT_FOUND
from app.rag.retrieval import RetrievedChunk
from evals.run_retrieval_eval import (
    QUESTIONS,
    CostGuard,
    EvalQuestion,
    evaluate_pipeline,
    normalize_for_match,
)
from tests.fakes import FakeLLM, FakeRetriever


@pytest.mark.parametrize(
    "labels",
    [
        {"answerable": True},
        {"answerable": True, "expected_doc_id": "leave", "expected_section": "Rule"},
        {
            "answerable": True,
            "expected_doc_id": "leave",
            "expected_section": "Rule",
            "expected_answer_contains": "  ",
        },
        {"answerable": False, "expected_doc_id": "leave"},
        {"answerable": False, "expected_answer_contains": "Fact"},
        {"answerable": False, "expected_answer_contains": ""},
    ],
)
def test_eval_rejects_inconsistent_labels(labels: dict) -> None:
    with pytest.raises(ValueError):
        EvalQuestion(id="bad", question="Question?", **labels)


def test_calibration_answer_phrases_are_exact_quotes_from_labelled_sections(
    settings: Settings,
) -> None:
    chunks = [
        chunk
        for document in load_documents(settings.corpus_dir)
        for chunk in chunk_document(document)
    ]
    questions = [
        EvalQuestion.model_validate_json(line)
        for line in QUESTIONS.read_text(encoding="utf-8").splitlines()
    ]
    assert len(questions) == 20
    assert sum(question.answerable for question in questions) == 15
    for question in questions:
        if question.answerable:
            phrase = question.expected_answer_contains
            assert phrase
            assert any(
                chunk.doc_id == question.expected_doc_id
                and chunk.section == question.expected_section
                and phrase in chunk.text
                for chunk in chunks
            ), question.id


@pytest.mark.parametrize(
    ("phrase", "answer"),
    [
        # The first three are forms gpt-oss-20b produced in the live smoke test.
        ("31 March", "Use them by 31 March."),
        ("SGD 220", "The nightly hotel limit is SGD 220 per room."),
        ("multi-factor authentication", "Multi‑factor authentication is required."),
        ("SGD 220", "Capped at SGD 220."),
        ("employee's manager", "Ask the employee’s manager."),
        ("up to 5 unused days", "You may carry over **up to 5** unused\ndays."),
    ],
)
def test_phrase_matching_ignores_llm_typography(phrase: str, answer: str) -> None:
    assert normalize_for_match(phrase) in normalize_for_match(answer)


@pytest.mark.parametrize(
    ("phrase", "answer"),
    [("SGD 220", "SGD 65 per day."), ("31 March", "31 May"), ("do not accumulate", "accumulate")],
)
def test_phrase_matching_still_rejects_different_wording(phrase: str, answer: str) -> None:
    assert normalize_for_match(phrase) not in normalize_for_match(answer)


def test_cost_guard_reserves_worst_case_and_refuses_over_budget() -> None:
    # 3 attempts x 500 completion tokens x $4e-6 = $0.006 per call; prompts cost nothing here.
    guard = CostGuard(0.01, 0.0, 4e-6, 500)
    assert guard.reserve(1000)
    assert not guard.reserve(1000)
    assert guard.reserved_usd == pytest.approx(0.006)


def test_pipeline_eval_scores_generation_and_keeps_raw_replies(
    settings: Settings, sources: list[RetrievedChunk]
) -> None:
    class DecliningLLM(FakeLLM):
        """Answers the supported question and declines the unsupported one."""

        async def generate(self, question: str, sources: list[RetrievedChunk]) -> LLMReply:
            if question == "Unsupported?":
                return LLMReply(text=NOT_FOUND)
            return await super().generate(question, sources)

    llm = DecliningLLM("Rule 1 applies [1].")
    questions = [
        EvalQuestion(
            id="supported",
            question="Which rule applies?",
            answerable=True,
            expected_doc_id="leave",
            expected_section=sources[0].chunk.section,
            expected_answer_contains="Rule 1 applies",
        ),
        EvalQuestion(id="unsupported", question="Unsupported?", answerable=False),
    ]
    result = asyncio.run(evaluate_pipeline(FakeRetriever(sources), llm, questions, settings))
    assert result["answer_contains_rate"] == 1
    assert result["citation_hit_rate"] == 1
    assert (result["hit@1"], result["hit@k"], result["mrr"]) == (1, 1, 1)
    assert result["unanswerable_refusal_rate"] == 1
    assert result["answerable_false_refusal_rate"] == 0
    assert result["routes"] == {"grounded": 1, "conversation": 1}
    assert result["refusal_reasons"] == {"model_not_found": 1}
    supported, unsupported = result["questions"]
    assert supported["raw_replies"] == ["Rule 1 applies [1]."]
    # The grounded step declined, then the source-free fallback declined too.
    assert unsupported["raw_replies"] == [NOT_FOUND, NOT_FOUND]


def test_pipeline_eval_skips_questions_the_budget_cannot_cover(
    settings: Settings, sources: list[RetrievedChunk]
) -> None:
    llm = FakeLLM("Rule 1 applies [1].")
    questions = [
        EvalQuestion(
            id=f"q{i}",
            question=f"Question {i}?",
            answerable=True,
            expected_doc_id="leave",
            expected_section=sources[0].chunk.section,
            expected_answer_contains="Rule 1 applies",
        )
        for i in range(2)
    ]
    questions.insert(1, EvalQuestion(id="u", question="Unsupported?", answerable=False))
    guard = CostGuard(0.025, 0.0, 4e-6, 500)  # each question reserves two $0.006 calls
    result = asyncio.run(evaluate_pipeline(FakeRetriever(sources), llm, questions, settings, guard))
    assert result["skipped_count"] == 1
    assert [row.get("skipped", False) for row in result["questions"]] == [False, False, True]
    assert len(llm.calls) + len(llm.conversations) == 2


def test_pipeline_eval_records_a_provider_failure_and_keeps_going(
    settings: Settings, sources: list[RetrievedChunk]
) -> None:
    class FlakyLLM(FakeLLM):
        async def generate(self, question: str, sources: list[RetrievedChunk]) -> LLMReply:
            if question == "Fails?":
                raise LLMError("APITimeoutError")
            return await super().generate(question, sources)

    questions = [
        EvalQuestion(
            id=question,
            question=question,
            answerable=True,
            expected_doc_id="leave",
            expected_section=sources[0].chunk.section,
            expected_answer_contains="Rule 1 applies",
        )
        for question in ["Fails?", "Works?"]
    ]
    questions.append(EvalQuestion(id="u", question="Unsupported?", answerable=False))
    llm = FlakyLLM("Rule 1 applies [1].")
    result = asyncio.run(evaluate_pipeline(FakeRetriever(sources), llm, questions, settings))
    assert result["errors"] == {"APITimeoutError": 1}
    assert result["answerable_count"] == 1
    assert result["questions"][0]["error"] == "APITimeoutError"


def test_pipeline_eval_counts_a_too_long_reply_as_a_refusal(
    settings: Settings, sources: list[RetrievedChunk]
) -> None:
    questions = [
        EvalQuestion(
            id="a",
            question="Whole policy?",
            answerable=True,
            expected_doc_id="leave",
            expected_section=sources[0].chunk.section,
            expected_answer_contains="Rule 1 applies",
        ),
        EvalQuestion(id="u", question="Unsupported?", answerable=False),
    ]
    llm = FakeLLM(truncate=frozenset({"generate"}))
    result = asyncio.run(evaluate_pipeline(FakeRetriever(sources), llm, questions, settings))
    row = result["questions"][0]
    assert (row["kind"], row["refused"], row["answer_contains"]) == ("too_long", True, False)
    assert result["answerable_false_refusal_rate"] == 1.0
    assert result["refusal_reasons"] == {"truncated": 2}


def test_pipeline_eval_records_an_embedding_failure_while_ranking(
    settings: Settings, sources: list[RetrievedChunk]
) -> None:
    class FlakyRetriever(FakeRetriever):
        def retrieve(self, question: str, top_k: int) -> list[RetrievedChunk]:
            if question == "Fails?":
                raise LLMError("APITimeoutError")
            return super().retrieve(question, top_k)

    questions = [
        EvalQuestion(
            id=question,
            question=question,
            answerable=True,
            expected_doc_id="leave",
            expected_section=sources[0].chunk.section,
            expected_answer_contains="Rule 1 applies",
        )
        for question in ["Fails?", "Works?"]
    ]
    questions.append(EvalQuestion(id="u", question="Unsupported?", answerable=False))
    llm = FakeLLM("Rule 1 applies [1].")
    result = asyncio.run(evaluate_pipeline(FlakyRetriever(sources), llm, questions, settings))
    assert result["errors"] == {"APITimeoutError": 1}
    assert result["questions"][0]["error"] == "APITimeoutError"
    assert result["answerable_count"] == 1
