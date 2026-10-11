"""Deterministic fakes and helpers: no network calls and no local credentials."""

from collections.abc import Sequence
from typing import Any

import numpy as np

from app.config import Settings
from app.rag.embeddings import VectorMatrix
from app.rag.llm import LLMReply, TokenUsage, TruncatedCompletion
from app.rag.prompts import NOT_FOUND
from app.rag.retrieval import RetrievedChunk
from app.schemas import Citation, RagAnswer


def make_settings(**values: Any) -> Settings:
    """Settings that never read a .env file, so a developer's real key can't leak in."""
    # pydantic-settings accepts _env_file at runtime; type checkers don't see the parameter.
    return Settings(_env_file=None, **values)  # pyright: ignore[reportCallIssue]


class FakeEmbedder:
    def __init__(self, documents: list[list[float]], query: list[float]) -> None:
        self.documents = documents
        self.query = query
        self.indexed_texts: list[str] = []
        self.queries: list[str] = []

    def embed_documents(self, texts: Sequence[str]) -> VectorMatrix:
        self.indexed_texts = list(texts)
        return np.asarray(self.documents, dtype=np.float32)

    def embed_query(self, text: str) -> VectorMatrix:
        self.queries.append(text)
        return np.asarray([self.query], dtype=np.float32)


class FakeRetriever:
    def __init__(self, results: list[RetrievedChunk]) -> None:
        self.results = results
        self.calls: list[tuple[str, int]] = []

    def retrieve(self, question: str, top_k: int) -> list[RetrievedChunk]:
        self.calls.append((question, top_k))
        return self.results[:top_k]


class FakeLLM:
    def __init__(
        self,
        text: str = "Employees get 22 days [1].",
        conversation_reply: str = NOT_FOUND,
        truncate: frozenset[str] = frozenset(),
    ) -> None:
        self.text = text
        self.conversation_reply = conversation_reply
        # Which calls ("generate", "converse") hit the token limit.
        self.truncate = truncate
        self.calls: list[tuple[str, list[RetrievedChunk]]] = []
        self.conversations: list[str] = []

    async def generate(self, question: str, sources: list[RetrievedChunk]) -> LLMReply:
        self.calls.append((question, sources))
        if "generate" in self.truncate:
            raise TruncatedCompletion(
                TokenUsage(prompt_tokens=7, completion_tokens=20, total_tokens=27)
            )
        return LLMReply(text=self.text)

    async def converse(self, question: str) -> LLMReply:
        self.conversations.append(question)
        if "converse" in self.truncate:
            raise TruncatedCompletion()
        return LLMReply(text=self.conversation_reply)


class FakePipeline:
    def __init__(self) -> None:
        self.questions: list[str] = []

    async def answer(self, question: str) -> RagAnswer:
        self.questions.append(question)
        return RagAnswer(
            answer="Employees get 14 days.",
            citations=[
                Citation(
                    doc_id="hr-1",
                    title="Leave Policy",
                    section="3. Leave",
                    snippet="14 days",
                    score=0.9,
                )
            ],
        )
