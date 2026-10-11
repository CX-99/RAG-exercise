"""Evaluate retrieval, refusals and answer content against the live hosted models."""

import argparse
import asyncio
import json
import re
import sys
import unicodedata
import urllib.request
from collections import Counter
from pathlib import Path
from time import perf_counter

from pydantic import BaseModel, model_validator

from app.config import Settings
from app.rag.chunking import chunk_document
from app.rag.documents import load_documents
from app.rag.embeddings import OpenRouterEmbedder
from app.rag.llm import LLMClient, LLMError, LLMReply, OpenRouterLLM
from app.rag.pipeline import PolicyPipeline, build_retriever
from app.rag.prompts import CONVERSATION_PROMPT, SYSTEM_PROMPT, source_prompt
from app.rag.retrieval import RetrievedChunk, Retriever
from app.rag.telemetry import RequestMetrics, request_metrics

QUESTIONS = Path(__file__).with_name("retrieval_questions.jsonl")
OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
OPENROUTER_EMBEDDING_MODELS_URL = "https://openrouter.ai/api/v1/embeddings/models"

# Typographic variants that models emit but hand-written labels don't use:
# U+2010-U+2015 hyphens and dashes, U+2212 minus, and curly single and double quotes.
_DASHES = re.compile("[‐-―−]")
_SINGLE_QUOTES = re.compile("[‘’‚‛]")
_DOUBLE_QUOTES = re.compile("[“”„‟]")
_MARKDOWN_MARKERS = re.compile(r"[*_`]")


def normalize_for_match(text: str) -> str:
    """Normalize text so a phrase check compares wording, not typography.

    Applies Unicode NFKC (which turns, e.g., a narrow no-break space into a space), maps
    dash and curly-quote variants to ASCII, drops Markdown emphasis and code markers,
    collapses whitespace, and casefolds. Apply it to both the phrase and the answer.
    """
    text = unicodedata.normalize("NFKC", text)
    text = _DASHES.sub("-", text)
    text = _SINGLE_QUOTES.sub("'", text)
    text = _DOUBLE_QUOTES.sub('"', text)
    text = _MARKDOWN_MARKERS.sub("", text)
    return re.sub(r"\s+", " ", text).strip().casefold()


class EvalQuestion(BaseModel):
    """One labelled question from a JSONL evaluation file."""

    id: str
    question: str
    answerable: bool
    expected_doc_id: str | None = None
    expected_section: str | None = None
    expected_answer_contains: str | None = None

    @model_validator(mode="after")
    def labels_match_answerability(self) -> "EvalQuestion":
        """Require all three labels on answerable questions and none on the others."""
        labels = (self.expected_doc_id, self.expected_section, self.expected_answer_contains)
        if self.answerable and not all(label and label.strip() for label in labels):
            raise ValueError(
                "Answerable questions need document, section, and answer phrase labels"
            )
        if not self.answerable and any(label is not None for label in labels):
            raise ValueError("Unanswerable questions must not have labels")
        return self


def _rank(sources: list[RetrievedChunk], question: EvalQuestion) -> int | None:
    """1-based position of the labelled section among the results, or None if absent."""
    return next(
        (
            i
            for i, source in enumerate(sources, start=1)
            if source.chunk.doc_id == question.expected_doc_id
            and source.chunk.section == question.expected_section
        ),
        None,
    )


class CostGuard:
    """Refuses any call whose worst-case cost could push total spend past a budget.

    The worst case assumes the SDK retries twice and every attempt uses the full completion
    limit, with prompt tokens estimated at three characters each (English averages about
    four, so this overestimates). Prices are OpenRouter's listed ones; the provider it
    routes to may charge more (about 1.9x in the live smoke test), which these assumptions
    absorb unless calls are also retried. Confirm real spend with the key's usage record.

    Args:
        max_usd: Budget for the whole run.
        prompt_usd_per_token: Listed input price.
        completion_usd_per_token: Listed output price; reasoning tokens are billed as output.
        max_completion_tokens: The ``LLM_MAX_TOKENS`` limit sent with each call.
        embedding_usd_per_token: Listed embedding price; embeddings have no output tokens.
    """

    ATTEMPTS = 3
    CHARS_PER_TOKEN = 3

    def __init__(
        self,
        max_usd: float,
        prompt_usd_per_token: float,
        completion_usd_per_token: float,
        max_completion_tokens: int,
        embedding_usd_per_token: float = 0.0,
    ) -> None:
        self.max_usd = max_usd
        self.prompt_usd_per_token = prompt_usd_per_token
        self.completion_usd_per_token = completion_usd_per_token
        self.max_completion_tokens = max_completion_tokens
        self.embedding_usd_per_token = embedding_usd_per_token
        self.reserved_usd = 0.0

    def reserve(self, *prompt_chars: int) -> bool:
        """Reserve the worst case of one call per prompt; False if that exceeds the budget."""
        worst = sum(
            self.ATTEMPTS
            * (
                chars / self.CHARS_PER_TOKEN * self.prompt_usd_per_token
                + self.max_completion_tokens * self.completion_usd_per_token
            )
            for chars in prompt_chars
        )
        return self._take(worst)

    def reserve_embedding(self, *text_chars: int) -> bool:
        """Reserve the worst case of embedding these texts; False if that exceeds the budget."""
        tokens = sum(text_chars) / self.CHARS_PER_TOKEN
        return self._take(self.ATTEMPTS * tokens * self.embedding_usd_per_token)

    def _take(self, worst: float) -> bool:
        if self.reserved_usd + worst > self.max_usd:
            return False
        self.reserved_usd += worst
        return True


def fetch_pricing(model: str, url: str = OPENROUTER_MODELS_URL) -> tuple[float, float]:
    """Return ``(prompt, completion)`` USD per token from one of OpenRouter's public lists.

    Chat models are listed at :data:`OPENROUTER_MODELS_URL` and embedding models at
    :data:`OPENROUTER_EMBEDDING_MODELS_URL`; embeddings have a zero completion price.

    Raises:
        ValueError: If the model isn't listed or has no price.
    """
    with urllib.request.urlopen(url, timeout=20) as response:
        models = json.load(response)["data"]
    for entry in models:
        if entry.get("id") == model:
            pricing = entry.get("pricing") or {}
            try:
                return float(pricing["prompt"]), float(pricing.get("completion") or 0)
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"No usable price listed for {model}") from error
    raise ValueError(f"Model not listed by OpenRouter: {model}")


class _RecordingLLM:
    """Delegates to a real client and keeps each raw reply, so discarded answers can be read."""

    def __init__(self, inner: LLMClient) -> None:
        self.inner = inner
        self.replies: list[str] = []

    async def generate(self, question: str, sources: list[RetrievedChunk]) -> LLMReply:
        reply = await self.inner.generate(question, sources)
        self.replies.append(reply.text)
        return reply

    async def converse(self, question: str) -> LLMReply:
        reply = await self.inner.converse(question)
        self.replies.append(reply.text)
        return reply


async def evaluate_pipeline(
    retriever: Retriever,
    llm: LLMClient,
    questions: list[EvalQuestion],
    settings: Settings,
    guard: CostGuard | None = None,
) -> dict:
    """Score retrieval and the full pipeline, generation included, on each question.

    Every question runs through :class:`PolicyPipeline` exactly as the API runs it, and its
    row keeps every raw model reply, so an answer discarded as uncited can be inspected.
    Retrieval metrics count every answerable question, including refused ones. With a
    guard, a question is skipped rather than risk exceeding the budget.

    Returns:
        Aggregate metrics plus one row per question.

    Raises:
        ValueError: If no answerable or no unanswerable question was evaluated.
    """
    recorder = _RecordingLLM(llm)
    pipeline = PolicyPipeline(retriever, recorder, settings)
    rows: list[dict] = []
    for question in questions:
        # Each question is embedded twice (here for its rank, and inside the pipeline) and
        # can make two calls: grounded generation, then the source-free fallback.
        if guard and not guard.reserve_embedding(len(question.question), len(question.question)):
            rows.append({"id": question.id, "answerable": question.answerable, "skipped": True})
            continue
        metrics = RequestMetrics()
        token = request_metrics.set(metrics)
        replies_before = len(recorder.replies)
        try:
            # Ranking embeds the question too, so it can fail like generation.
            sources = await asyncio.to_thread(
                retriever.retrieve, question.question, settings.retrieval_top_k
            )
            grounded_chars = len(SYSTEM_PROMPT) + len(source_prompt(question.question, sources))
            fallback_chars = len(CONVERSATION_PROMPT) + len(question.question)
            if guard and not guard.reserve(grounded_chars, fallback_chars):
                rows.append({"id": question.id, "answerable": question.answerable, "skipped": True})
                continue
            result = await pipeline.answer(question.question)
        except LLMError as error:
            # One provider failure must not discard the rest of a paid run.
            rows.append(
                {
                    "id": question.id,
                    "question": question.question,
                    "answerable": question.answerable,
                    "error": error.kind,
                    "provider_status": error.status_code,
                    "raw_replies": recorder.replies[replies_before:],
                }
            )
            continue
        finally:
            request_metrics.reset(token)
        # A too-long reply isn't shown either, so it counts as a refusal.
        refused = result.kind in ("not_found", "too_long")
        contains = None
        if question.answerable:
            phrase = question.expected_answer_contains
            assert phrase is not None  # EvalQuestion requires it for answerable questions
            contains = not refused and normalize_for_match(phrase) in normalize_for_match(
                result.answer
            )
        cited = [{"doc_id": c.doc_id, "section": c.section} for c in result.citations]
        expected = {"doc_id": question.expected_doc_id, "section": question.expected_section}
        rows.append(
            {
                "id": question.id,
                "question": question.question,
                "answerable": question.answerable,
                "rank": _rank(sources, question),
                "best_dense": max((source.dense_score for source in sources), default=-1.0),
                "route": metrics.route,
                "kind": result.kind,
                "refused": refused,
                "refusal_reason": metrics.refusal_reason,
                "answer": result.answer,
                "raw_replies": recorder.replies[replies_before:],
                "cited": cited,
                "cites_expected_section": expected in cited if question.answerable else None,
                "expected_answer_contains": question.expected_answer_contains,
                "answer_contains": contains,
                "usage": metrics.usage.model_dump() if metrics.usage else None,
                "generation_ms": round(metrics.generation_ms, 1),
            }
        )
    # Rates cover completed questions; skipped and failed ones are counted separately.
    evaluated = [row for row in rows if not row.get("skipped") and not row.get("error")]
    answerable = [row for row in evaluated if row["answerable"]]
    unanswerable = [row for row in evaluated if not row["answerable"]]
    if not answerable or not unanswerable:
        raise ValueError("Evaluation needs answerable and unanswerable questions")
    usages = [row["usage"] for row in evaluated if row["usage"]]
    return {
        "llm_model": settings.llm_model,
        "embedding_model": settings.embedding_model,
        "top_k": settings.retrieval_top_k,
        "answerable_count": len(answerable),
        "unanswerable_count": len(unanswerable),
        "skipped_count": sum(bool(row.get("skipped")) for row in rows),
        "errors": dict(Counter(row["error"] for row in rows if row.get("error"))),
        "hit@1": sum(row["rank"] == 1 for row in answerable) / len(answerable),
        "hit@k": sum(row["rank"] is not None for row in answerable) / len(answerable),
        "mrr": sum(1 / row["rank"] for row in answerable if row["rank"]) / len(answerable),
        "answer_contains_rate": sum(row["answer_contains"] for row in answerable) / len(answerable),
        "citation_hit_rate": sum(row["cites_expected_section"] for row in answerable)
        / len(answerable),
        "unanswerable_refusal_rate": sum(row["refused"] for row in unanswerable)
        / len(unanswerable),
        "answerable_false_refusal_rate": sum(row["refused"] for row in answerable)
        / len(answerable),
        "refusal_reasons": dict(Counter(r["refusal_reason"] for r in evaluated if r["refused"])),
        "routes": dict(Counter(row["route"] for row in evaluated)),
        "prompt_tokens": sum(usage["prompt_tokens"] for usage in usages),
        "completion_tokens": sum(usage["completion_tokens"] for usage in usages),
        "questions": rows,
    }


def _corpus_chars(settings: Settings) -> int:
    """Characters the index embeds, so the guard can reserve the indexing cost first."""
    documents = load_documents(settings.corpus_dir)
    return sum(len(chunk.indexed_text) for doc in documents for chunk in chunk_document(doc))


async def main() -> int:
    """Command-line entry point: a live, budget-capped run that reads the key from .env.

    Returns a process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, default=QUESTIONS)
    parser.add_argument("--output", type=Path, help="Optional JSON report path")
    parser.add_argument("--max-cost-usd", type=float, default=0.03, help="Budget cap for the run")
    args = parser.parse_args()
    questions = [
        EvalQuestion.model_validate_json(line)
        for line in args.questions.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    started = perf_counter()
    settings = Settings()
    prompt_price, completion_price = fetch_pricing(settings.llm_model)
    embedding_price, _ = fetch_pricing(settings.embedding_model, OPENROUTER_EMBEDDING_MODELS_URL)
    guard = CostGuard(
        args.max_cost_usd,
        prompt_price,
        completion_price,
        settings.llm_max_tokens,
        embedding_price,
    )
    if not guard.reserve_embedding(_corpus_chars(settings)):
        print("Budget too small to index the corpus", file=sys.stderr)
        return 1
    retriever = await asyncio.to_thread(build_retriever, settings)
    llm = OpenRouterLLM(settings)
    try:
        result = await evaluate_pipeline(retriever, llm, questions, settings, guard)
    finally:
        await llm.aclose()
        if isinstance(retriever.embedder, OpenRouterEmbedder):
            retriever.embedder.close()
    result["prices_usd_per_token"] = {
        "prompt": prompt_price,
        "completion": completion_price,
        "embedding": embedding_price,
    }
    result["budget_usd"] = args.max_cost_usd
    result["worst_case_reserved_usd"] = round(guard.reserved_usd, 6)
    result["token_priced_generation_cost_usd"] = round(
        result["prompt_tokens"] * prompt_price + result["completion_tokens"] * completion_price, 7
    )
    result["elapsed_seconds"] = round(perf_counter() - started, 3)
    report = json.dumps(result, indent=2)
    print(report)
    if args.output:
        args.output.write_text(report + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
