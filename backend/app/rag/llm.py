"""Hosted generation through OpenRouter, behind a protocol that tests can fake."""

from typing import Protocol

import httpx2
from openai import APIError, AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel

from app.config import Settings
from app.rag.errors import LLMError
from app.rag.prompts import CONVERSATION_PROMPT, SYSTEM_PROMPT, source_prompt
from app.rag.retrieval import RetrievedChunk

__all__ = [
    "LLMClient",
    "LLMError",
    "LLMReply",
    "OpenRouterLLM",
    "TokenUsage",
    "TruncatedCompletion",
]


class TokenUsage(BaseModel):
    """Token counts reported for one completion; ``completion_tokens`` includes reasoning."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class TruncatedCompletion(LLMError):
    """The reply hit the completion-token limit before it finished.

    A cut-off reply can lose a qualification or a citation, so it is never shown. The
    pipeline turns this into a ``too_long`` response rather than a server error.

    Attributes:
        usage: Token counts for the cut-off call, if reported; they are still billed.
    """

    def __init__(self, usage: TokenUsage | None = None) -> None:
        super().__init__("TruncatedCompletion")
        self.usage = usage


class LLMReply(BaseModel):
    """Raw model output and, when the provider reports it, token usage."""

    text: str
    usage: TokenUsage | None = None


class LLMClient(Protocol):
    """Produces a reply from a question and its numbered sources, or without sources."""

    async def generate(self, question: str, sources: list[RetrievedChunk]) -> LLMReply:
        """Return a reply that cites sources as ``[n]`` (1-based), or the not-found text."""
        ...

    async def converse(self, question: str) -> LLMReply:
        """Reply to a message that matched no policy section, or return the not-found text."""
        ...


class OpenRouterLLM:
    """Grounded generation through OpenRouter's OpenAI-compatible chat completions API.

    Args:
        settings: Supplies the key, base URL, model, token limit and timeout.
        http_client: Optional HTTP client; tests inject one with a mock transport.

    Raises:
        ValueError: If the API key is missing or blank.
    """

    def __init__(self, settings: Settings, http_client: httpx2.AsyncClient | None = None) -> None:
        key = settings.openrouter_api_key
        if key is None or not key.get_secret_value().strip():
            raise ValueError("OPENROUTER_API_KEY is required for generation")
        self._model = settings.llm_model
        self._max_tokens = settings.llm_max_tokens
        self._client = AsyncOpenAI(
            api_key=key.get_secret_value(),
            base_url=settings.openrouter_base_url,
            timeout=settings.llm_timeout_seconds,
            max_retries=2,
            http_client=http_client,
        )

    async def generate(self, question: str, sources: list[RetrievedChunk]) -> LLMReply:
        """Ask the model to answer from the numbered sources only.

        Returns:
            The reply text, with token usage, or ``None`` usage if any count is missing.

        Raises:
            TruncatedCompletion: If the reply is cut off by the token limit.
            LLMError: On a provider error, a timeout after the SDK's retries, or an empty
                reply.
        """
        return await self._complete(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": source_prompt(question, sources)},
            ]
        )

    async def converse(self, question: str) -> LLMReply:
        """Reply to a message that matched no policy section, with no sources attached.

        Greetings and questions about the assistant get a short reply; anything else gets
        the not-found text.

        Raises:
            TruncatedCompletion: If the reply is cut off by the token limit.
            LLMError: As for :meth:`generate`.
        """
        return await self._complete(
            [
                {"role": "system", "content": CONVERSATION_PROMPT},
                {"role": "user", "content": question},
            ]
        )

    async def _complete(self, messages: list[ChatCompletionMessageParam]) -> LLMReply:
        """Run one chat completion and map every failure to :class:`LLMError`."""
        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                temperature=0,
                max_tokens=self._max_tokens,
                messages=messages,
            )
        except (APIError, httpx2.HTTPError, TimeoutError) as error:
            raise LLMError(type(error).__name__, getattr(error, "status_code", None)) from error
        if not response.choices:
            raise LLMError("EmptyCompletion")
        choice = response.choices[0]
        usage = _token_usage(response.usage)
        # The limit also covers the model's hidden reasoning, which counts as completion.
        if choice.finish_reason == "length":
            raise TruncatedCompletion(usage)
        text = choice.message.content
        if not text or not text.strip():
            raise LLMError("EmptyCompletion")
        return LLMReply(text=text, usage=usage)

    async def aclose(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.close()


def _token_usage(usage: object) -> TokenUsage | None:
    """Read the provider's token counts; ``None`` if any count is missing or null."""
    prompt = getattr(usage, "prompt_tokens", None)
    completion = getattr(usage, "completion_tokens", None)
    total = getattr(usage, "total_tokens", None)
    if prompt is None or completion is None or total is None:
        return None
    return TokenUsage(prompt_tokens=prompt, completion_tokens=completion, total_tokens=total)
