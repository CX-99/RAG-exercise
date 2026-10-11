"""Request and response models that define the public API contract.

frontend/src/types.ts mirrors these models; keep the two in sync.
"""

from typing import Annotated, Literal

from pydantic import BaseModel, Field, StringConstraints

MAX_QUESTION_CHARS = 2000

# Whitespace is trimmed before the length check, so "   " is rejected as empty.
Question = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_QUESTION_CHARS)
]


class HealthResponse(BaseModel):
    """Liveness payload returned by ``GET /api/health``."""

    status: str
    version: str


class AskRequest(BaseModel):
    """Body of ``POST /api/ask``: one question of 1-2000 characters after trimming."""

    question: Question


class Citation(BaseModel):
    """A source passage the answer is grounded in.

    Attributes:
        doc_id: Corpus file stem, e.g. ``annual-leave``.
        title: Document title.
        section: Heading path, e.g. ``3. Annual Leave > 3.2 Carry-over``.
        snippet: Raw section text, without the title/section prefix used for indexing.
        score: Dense cosine similarity to the question; not a probability.
    """

    doc_id: str
    title: str
    section: str
    snippet: str
    score: float


# What a response is, so clients can label it without parsing its text.
AnswerKind = Literal["answer", "general", "chat", "not_found", "too_long"]


class RagAnswer(BaseModel):
    """What a RAG pipeline produces, independent of HTTP concerns.

    Attributes:
        answer: The response text, with ``[n]`` markers for any citations.
        kind: ``answer`` (generated from cited sources), ``general`` (labelled general
            knowledge, no sources), ``chat`` (small talk), ``not_found`` (declined), or
            ``too_long`` (the reply hit the token limit, so it was not shown).
        citations: The sources the answer cites, numbered by first appearance.
    """

    answer: str
    kind: AnswerKind = "answer"
    citations: list[Citation] = Field(default_factory=list)


class AskResponse(RagAnswer):
    """A pipeline answer plus the request ID that also appears in the server log."""

    request_id: str
