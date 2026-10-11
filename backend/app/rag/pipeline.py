"""Pipeline orchestration and the lazily built, process-wide pipeline instance."""

import asyncio
import re
from time import perf_counter
from typing import Protocol

from app.config import Settings, get_settings
from app.rag.chunking import chunk_document
from app.rag.citations import (
    CITATION_GROUP,
    is_not_found,
    normalize_brackets,
    process_citations,
)
from app.rag.documents import load_documents
from app.rag.embeddings import OpenRouterEmbedder
from app.rag.llm import LLMClient, OpenRouterLLM, TokenUsage, TruncatedCompletion
from app.rag.prompts import GENERAL_KNOWLEDGE_PREFIX, NOT_FOUND
from app.rag.retrieval import HybridRetriever, Retriever
from app.rag.telemetry import RefusalReason, RequestMetrics, request_metrics
from app.schemas import RagAnswer

# The model sometimes formats the label, e.g. "**General knowledge (...):**".
_GENERAL_LABEL = re.compile(
    r"[*_]*\s*" + re.escape(GENERAL_KNOWLEDGE_PREFIX) + r"\s*[*_]*\s*", re.IGNORECASE
)

# Returned instead of a cut-off reply; the UI shows its own wording for this kind.
TOO_LONG = (
    "This question needs a longer answer than I can give in one reply. "
    "Please ask about one part of a policy instead."
)


class RagPipeline(Protocol):
    """Turns a question into a grounded answer, independently of HTTP."""

    async def answer(self, question: str) -> RagAnswer:
        """Return an answer with its citations, or the not-found answer."""
        ...


def _add_usage(first: TokenUsage | None, second: TokenUsage | None) -> TokenUsage | None:
    """Sum the token usage of two calls, keeping whichever side is known."""
    if first is None or second is None:
        return first or second
    return TokenUsage(
        prompt_tokens=first.prompt_tokens + second.prompt_tokens,
        completion_tokens=first.completion_tokens + second.completion_tokens,
        total_tokens=first.total_tokens + second.total_tokens,
    )


def _too_long(metrics: RequestMetrics, error: TruncatedCompletion) -> RagAnswer:
    """Decline a reply that hit the token limit, still counting the tokens it used."""
    metrics.usage = _add_usage(metrics.usage, error.usage)
    metrics.refused = True
    metrics.refusal_reason = "truncated"
    return RagAnswer(answer=TOO_LONG, kind="too_long")


class PolicyPipeline:
    """Retrieve, generate from sources, validate citations, then fall back if needed.

    Timings and refusal details go to the current request's
    :class:`~app.rag.telemetry.RequestMetrics`, if the caller set one.

    Args:
        retriever: Ranks corpus chunks for a question.
        llm: Generates a reply from the retrieved sources.
        settings: Supplies ``RETRIEVAL_TOP_K``.
    """

    def __init__(self, retriever: Retriever, llm: LLMClient, settings: Settings) -> None:
        self.retriever = retriever
        self.llm = llm
        self.top_k = settings.retrieval_top_k

    async def answer(self, question: str) -> RagAnswer:
        """Answer ``question`` from the corpus, or fall back to a reply without sources.

        The model always gets the retrieved sources first, and decides whether they answer
        the question. If it gives no cited answer, it gets the message alone: it may make
        small talk or give a labelled general-knowledge answer, but must decline questions
        about the company's own rules. A reply cut off by the token limit is never shown;
        the answer is ``too_long`` instead.

        Raises:
            LLMError: If embedding or generation fails.
        """
        metrics = request_metrics.get() or RequestMetrics()
        started = perf_counter()
        try:
            sources = await asyncio.to_thread(self.retriever.retrieve, question, self.top_k)
        finally:
            metrics.retrieval_ms = (perf_counter() - started) * 1000
        metrics.chunk_count = len(sources)
        metrics.route = "grounded"
        started = perf_counter()
        try:
            reply = await self.llm.generate(question, sources)
        except TruncatedCompletion as error:
            return _too_long(metrics, error)
        finally:
            metrics.generation_ms = (perf_counter() - started) * 1000
        metrics.usage = reply.usage
        result = process_citations(reply.text, sources)
        if result.answer != NOT_FOUND:
            return result
        reason: RefusalReason = "model_not_found" if is_not_found(reply.text) else "uncited"
        return await self._converse(question, metrics, refusal_reason=reason)

    async def _converse(
        self, question: str, metrics: RequestMetrics, refusal_reason: RefusalReason
    ) -> RagAnswer:
        """Get a source-free reply: small talk, a labelled general answer, or a decline."""
        # Set first, so a failure in this call isn't logged as a grounded one.
        metrics.route = "conversation"
        started = perf_counter()
        try:
            reply = await self.llm.converse(question)
        except TruncatedCompletion as error:
            return _too_long(metrics, error)
        finally:
            metrics.generation_ms += (perf_counter() - started) * 1000
        metrics.usage = _add_usage(metrics.usage, reply.usage)
        # No sources were shown, so any [n] marker in the reply would point at nothing.
        text = CITATION_GROUP.sub("", normalize_brackets(reply.text)).strip()
        label = _GENERAL_LABEL.match(text)
        body = text[label.end() :].strip() if label else text
        if not body or is_not_found(body):
            metrics.refused = True
            metrics.refusal_reason = refusal_reason
            return RagAnswer(answer=NOT_FOUND, kind="not_found")
        if label:
            # Rewritten with the exact label, which the UI replaces with its badge.
            metrics.route = "general"
            return RagAnswer(answer=f"{GENERAL_KNOWLEDGE_PREFIX} {body}", kind="general")
        return RagAnswer(answer=text, kind="chat")


def build_retriever(settings: Settings) -> HybridRetriever:
    """Load, chunk and index the corpus; indexing embeds every chunk through the API.

    Raises:
        ValueError: If the corpus directory is missing, has duplicate stems, or yields no
            chunks.
        LLMError: If the embeddings API fails.
    """
    documents = load_documents(settings.corpus_dir)
    chunks = [chunk for document in documents for chunk in chunk_document(document)]
    if not chunks:
        raise ValueError("Corpus contains no non-empty chunks")
    return HybridRetriever(chunks, OpenRouterEmbedder(settings))


def build_pipeline(settings: Settings) -> PolicyPipeline:
    """Build the pipeline that ``settings`` describes."""
    return PolicyPipeline(build_retriever(settings), OpenRouterLLM(settings), settings)


class PipelineUnavailable(Exception):
    """Construction failed; retain only the exception type for safe diagnostics."""

    def __init__(self, kind: str) -> None:
        super().__init__("Policy assistant is temporarily unavailable.")
        self.kind = kind


# One pipeline per process, built on the first request.
_pipeline: PolicyPipeline | None = None
_build_lock = asyncio.Lock()
# Failed builds so far, and the last one's exception type. Requests that were waiting
# while a build failed fail with it, rather than each starting another slow build.
_build_failures = 0
_last_failure_kind = ""


async def get_pipeline() -> RagPipeline:
    """FastAPI dependency that returns the shared pipeline, building it on first use.

    A built pipeline is returned without locking. Concurrent first requests share one
    build, which runs in a worker thread so the event loop stays responsive. If it fails,
    the requests waiting on it fail too. The failure is not cached, so a request that
    arrives afterwards retries the build.

    Raises:
        PipelineUnavailable: If construction fails; only the exception type is kept.
    """
    global _pipeline, _build_failures, _last_failure_kind
    if _pipeline is not None:
        return _pipeline
    failures_before_waiting = _build_failures
    async with _build_lock:
        if _pipeline is not None:
            return _pipeline
        if _build_failures != failures_before_waiting:
            raise PipelineUnavailable(_last_failure_kind)
        try:
            _pipeline = await asyncio.to_thread(build_pipeline, get_settings())
        except Exception as error:
            _build_failures += 1
            _last_failure_kind = type(error).__name__
            raise PipelineUnavailable(_last_failure_kind) from None
        return _pipeline
