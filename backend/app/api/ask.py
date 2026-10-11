"""Question-answering endpoint: request IDs, error mapping, and request logs."""

import logging
import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from app.rag.llm import LLMError
from app.rag.pipeline import RagPipeline, get_pipeline
from app.rag.telemetry import RequestMetrics, request_metrics
from app.schemas import AskRequest, AskResponse

logger = logging.getLogger(__name__)
router = APIRouter(tags=["ask"])


@router.post("/ask")
async def ask(
    request: AskRequest,
    pipeline: Annotated[RagPipeline, Depends(get_pipeline)],
) -> AskResponse:
    """Answer a policy question with cited sources.

    Returns 503 if a hosted model (embeddings or generation) fails or times out. Each
    request writes one log line with timings and refusal details, but never the question
    text.
    """
    request_id = uuid.uuid4().hex
    metrics = RequestMetrics()
    token = request_metrics.set(metrics)
    error_kind, provider_status = "none", None
    try:
        result = await pipeline.answer(request.question)
        return AskResponse(request_id=request_id, **result.model_dump())
    except (LLMError, TimeoutError) as error:
        error_kind = error.kind if isinstance(error, LLMError) else type(error).__name__
        provider_status = getattr(error, "status_code", None)
        raise HTTPException(
            status_code=503, detail="Answer generation is temporarily unavailable."
        ) from None
    except Exception as error:
        error_kind = type(error).__name__
        raise
    finally:
        # Provider exception messages/bodies can echo the question; log only safe diagnostics.
        logger.log(
            logging.ERROR if error_kind != "none" else logging.INFO,
            "ask request_id=%s retrieval_ms=%.2f generation_ms=%.2f chunk_count=%d route=%s "
            "refused=%s refusal_reason=%s token_usage=%s error_type=%s provider_status=%s",
            request_id,
            metrics.retrieval_ms,
            metrics.generation_ms,
            metrics.chunk_count,
            metrics.route,
            "yes" if metrics.refused else "no",
            metrics.refusal_reason,
            metrics.usage.model_dump() if metrics.usage else None,
            error_kind,
            provider_status,
        )
        request_metrics.reset(token)
