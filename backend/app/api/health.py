"""Liveness endpoint."""

from fastapi import APIRouter

from app import __version__
from app.schemas import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health")
async def health() -> HealthResponse:
    """Report that the process is up; this does not check model or index readiness."""
    return HealthResponse(status="ok", version=__version__)
