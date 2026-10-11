import asyncio
import threading

import pytest

from app.config import Settings
from app.rag.llm import LLMError, LLMReply, TokenUsage
from app.rag.pipeline import TOO_LONG, PolicyPipeline
from app.rag.prompts import GENERAL_KNOWLEDGE_PREFIX, NOT_FOUND
from app.rag.retrieval import RetrievedChunk
from app.rag.telemetry import RequestMetrics, request_metrics
from app.schemas import RagAnswer
from tests.fakes import FakeLLM, FakeRetriever


def run(pipeline: PolicyPipeline, question: str) -> tuple[RagAnswer, RequestMetrics]:
    metrics = RequestMetrics()
    token = request_metrics.set(metrics)
    try:
        return asyncio.run(pipeline.answer(question)), metrics
    finally:
        request_metrics.reset(token)


def test_every_question_is_first_answered_from_its_sources(
    settings: Settings, sources: list[RetrievedChunk]
) -> None:
    # Even weakly similar sources reach the model, which decides whether they answer.
    low = [source.model_copy(update={"dense_score": 0.1}) for source in sources]
    llm = FakeLLM("Use the VPN [1].")
    result, metrics = run(PolicyPipeline(FakeRetriever(low), llm, settings), "VPN?")
    assert result.answer == "Use the VPN [1]."
    assert result.kind == "answer"
    assert llm.calls == [("VPN?", low)] and llm.conversations == []
    assert metrics.route == "grounded" and not metrics.refused


def test_grounded_citations_are_renumbered_from_the_sources_shown(
    settings: Settings, sources: list[RetrievedChunk]
) -> None:
    llm = FakeLLM("Supported by the second source [2].")
    result, _ = run(PolicyPipeline(FakeRetriever(sources), llm, settings), "Question?")
    assert result.answer == "Supported by the second source [1]."
    assert result.citations[0].section == sources[1].chunk.section


def test_small_talk_falls_back_to_a_reply_without_sources(
    settings: Settings, sources: list[RetrievedChunk]
) -> None:
    llm = FakeLLM(NOT_FOUND, conversation_reply="Hello! Try asking about annual leave. [1]")
    result, metrics = run(PolicyPipeline(FakeRetriever(sources), llm, settings), "hi")
    # The reply had no sources, so the stray marker is removed and nothing is cited.
    assert result.answer == "Hello! Try asking about annual leave."
    assert result.kind == "chat" and result.citations == []
    assert len(llm.calls) == 1 and llm.conversations == ["hi"]
    assert metrics.route == "conversation"
    assert not metrics.refused and metrics.refusal_reason is None


@pytest.mark.parametrize("grounded_reply", [NOT_FOUND, "An uncited claim."])
def test_unanswered_grounded_reply_falls_back_to_labelled_general_knowledge(
    settings: Settings, sources: list[RetrievedChunk], grounded_reply: str
) -> None:
    general = f"{GENERAL_KNOWLEDGE_PREFIX} A VPN encrypts traffic between two networks."
    llm = FakeLLM(grounded_reply, conversation_reply=general)
    result, metrics = run(PolicyPipeline(FakeRetriever(sources), llm, settings), "What is a VPN?")
    assert result.answer == general and result.citations == []
    assert result.kind == "general"
    assert len(llm.calls) == 1 and llm.conversations == ["What is a VPN?"]
    assert metrics.route == "general" and not metrics.refused


@pytest.mark.parametrize(
    ("grounded_reply", "reason"),
    [
        (NOT_FOUND + " [1, 2]", "model_not_found"),
        ("A claim without evidence.", "uncited"),
        ("A claim with invalid evidence [99].", "uncited"),
    ],
)
def test_company_question_without_an_answer_is_declined_with_the_grounded_reason(
    settings: Settings, sources: list[RetrievedChunk], grounded_reply: str, reason: str
) -> None:
    llm = FakeLLM(grounded_reply)  # the fallback declines too
    result, metrics = run(PolicyPipeline(FakeRetriever(sources), llm, settings), "Sick days?")
    assert result.answer == NOT_FOUND and result.citations == []
    assert result.kind == "not_found"
    assert metrics.refused and metrics.refusal_reason == reason
    assert metrics.route == "conversation"
    assert len(llm.calls) == 1 and len(llm.conversations) == 1


def test_a_fallback_reply_of_only_markers_is_a_decline(
    settings: Settings, sources: list[RetrievedChunk]
) -> None:
    llm = FakeLLM(NOT_FOUND, conversation_reply=" [1] ")
    result, _ = run(PolicyPipeline(FakeRetriever(sources), llm, settings), "?")
    assert result.answer == NOT_FOUND and result.kind == "not_found"


def test_retrieval_runs_on_worker_thread_and_records_metrics(
    settings: Settings, sources: list[RetrievedChunk]
) -> None:
    threads = []

    class ThreadRecordingRetriever(FakeRetriever):
        def retrieve(self, question: str, top_k: int) -> list[RetrievedChunk]:
            threads.append(threading.get_ident())
            return super().retrieve(question, top_k)

    pipeline = PolicyPipeline(ThreadRecordingRetriever(sources), FakeLLM(), settings)
    _, metrics = run(pipeline, "Question?")
    assert threads[0] != threading.get_ident()
    assert metrics.chunk_count == 3
    assert metrics.retrieval_ms > 0
    assert metrics.generation_ms > 0
    assert metrics.route == "grounded" and not metrics.refused


def test_token_usage_is_summed_across_grounded_and_fallback_calls(
    settings: Settings, sources: list[RetrievedChunk]
) -> None:
    usage = TokenUsage(prompt_tokens=10, completion_tokens=2, total_tokens=12)

    class MeteredLLM(FakeLLM):
        async def generate(self, question: str, sources: list[RetrievedChunk]) -> LLMReply:
            return LLMReply(text=NOT_FOUND, usage=usage)

        async def converse(self, question: str) -> LLMReply:
            return LLMReply(text="Hello!", usage=usage)

    _, metrics = run(PolicyPipeline(FakeRetriever(sources), MeteredLLM(), settings), "hi")
    assert metrics.usage == TokenUsage(prompt_tokens=20, completion_tokens=4, total_tokens=24)
    assert metrics.route == "conversation"


def test_a_grounded_reply_cut_off_by_the_token_limit_is_too_long(
    settings: Settings, sources: list[RetrievedChunk]
) -> None:
    llm = FakeLLM(truncate=frozenset({"generate"}))
    result, metrics = run(PolicyPipeline(FakeRetriever(sources), llm, settings), "Whole policy?")
    # The partial reply is not shown, and the question doesn't go to the fallback.
    assert result == RagAnswer(answer=TOO_LONG, kind="too_long")
    assert llm.conversations == []
    assert metrics.route == "grounded"
    assert metrics.refused and metrics.refusal_reason == "truncated"
    # The cut-off call is still billed, so its tokens are still logged.
    assert metrics.usage == TokenUsage(prompt_tokens=7, completion_tokens=20, total_tokens=27)


def test_a_fallback_reply_cut_off_by_the_token_limit_is_too_long(
    settings: Settings, sources: list[RetrievedChunk]
) -> None:
    llm = FakeLLM(NOT_FOUND, truncate=frozenset({"converse"}))
    result, metrics = run(PolicyPipeline(FakeRetriever(sources), llm, settings), "Long essay?")
    assert result.kind == "too_long" and result.citations == []
    assert metrics.route == "conversation"
    assert metrics.refused and metrics.refusal_reason == "truncated"


@pytest.mark.parametrize(
    "label",
    [
        GENERAL_KNOWLEDGE_PREFIX,
        f"**{GENERAL_KNOWLEDGE_PREFIX}**",
        GENERAL_KNOWLEDGE_PREFIX.upper(),
    ],
)
def test_a_formatted_general_knowledge_label_is_still_recognised(
    settings: Settings, sources: list[RetrievedChunk], label: str
) -> None:
    llm = FakeLLM(NOT_FOUND, conversation_reply=f"{label} Paris is the capital of France.")
    result, metrics = run(PolicyPipeline(FakeRetriever(sources), llm, settings), "Capital?")
    assert result.kind == "general" and metrics.route == "general"
    # Rewritten with the exact label, which the UI replaces with its badge.
    assert result.answer == f"{GENERAL_KNOWLEDGE_PREFIX} Paris is the capital of France."


def test_a_label_with_no_answer_is_a_decline(
    settings: Settings, sources: list[RetrievedChunk]
) -> None:
    llm = FakeLLM(NOT_FOUND, conversation_reply=f"**{GENERAL_KNOWLEDGE_PREFIX}** {NOT_FOUND}")
    result, metrics = run(PolicyPipeline(FakeRetriever(sources), llm, settings), "Sick days?")
    assert result.kind == "not_found"
    assert metrics.refused and metrics.refusal_reason == "model_not_found"


def test_a_failed_fallback_call_is_logged_as_the_fallback(
    settings: Settings, sources: list[RetrievedChunk]
) -> None:
    class FailingFallback(FakeLLM):
        async def converse(self, question: str) -> LLMReply:
            raise LLMError("RateLimitError", 429)

    metrics = RequestMetrics()
    token = request_metrics.set(metrics)
    try:
        pipeline = PolicyPipeline(FakeRetriever(sources), FailingFallback(NOT_FOUND), settings)
        with pytest.raises(LLMError):
            asyncio.run(pipeline.answer("hi"))
    finally:
        request_metrics.reset(token)
    assert metrics.route == "conversation"
