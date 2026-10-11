"""Per-request metrics without changing the RagPipeline API or logging user text."""

from contextvars import ContextVar
from dataclasses import dataclass
from typing import Literal

from app.rag.llm import TokenUsage

# Why a request got no answer from the documents; see RequestMetrics.refusal_reason.
RefusalReason = Literal["model_not_found", "uncited", "truncated"]


@dataclass
class RequestMetrics:
    """Measurements for one request, filled in by the pipeline and logged by the route."""

    retrieval_ms: float = 0.0
    generation_ms: float = 0.0
    chunk_count: int = 0
    refused: bool = False
    # "truncated": the reply hit the token limit, so the answer is ``too_long``.
    refusal_reason: RefusalReason | None = None
    # "grounded": answered from cited sources; "conversation": small talk or a decline
    # without sources; "general": a labelled general-knowledge answer without sources.
    route: Literal["grounded", "conversation", "general"] | None = None
    usage: TokenUsage | None = None


# Set by the /api/ask route for the duration of one request. A context variable is
# isolated per request, and it reaches worker threads started with asyncio.to_thread.
request_metrics: ContextVar[RequestMetrics | None] = ContextVar("rag_metrics", default=None)
