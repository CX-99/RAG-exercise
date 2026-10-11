"""Application factory: logging, routes, and construction-error handling."""

import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app import __version__
from app.api import ask, health
from app.config import get_settings
from app.rag.pipeline import PipelineUnavailable

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    """Build the FastAPI application.

    Settings are validated here, so a misconfigured provider stops the server at startup
    instead of failing on the first request.
    """
    settings = get_settings()
    logging.basicConfig(
        level=settings.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # SDK debug logs can include request bodies; policy questions must never be logged.
    logging.getLogger("openai").setLevel(logging.WARNING)

    app = FastAPI(title="Policy RAG API", version=__version__)

    @app.exception_handler(PipelineUnavailable)
    async def pipeline_unavailable(request: Request, error: PipelineUnavailable) -> JSONResponse:
        """Return a generic 503 and log only the error type, never its message."""
        logger.error("pipeline_construction error_type=%s", error.kind)
        return JSONResponse(
            status_code=503, content={"detail": "Policy assistant is temporarily unavailable."}
        )

    app.include_router(health.router, prefix="/api")
    app.include_router(ask.router, prefix="/api")
    return app


app = create_app()
