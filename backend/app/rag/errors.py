"""Failures of the hosted model APIs, shared by embeddings and generation."""


class LLMError(Exception):
    """A hosted-model failure (embedding or generation) with safe diagnostics only.

    The message never includes provider response content, which can echo the question.

    Attributes:
        kind: The underlying exception class or failure name, e.g. ``TruncatedCompletion``.
        status_code: The provider's HTTP status, if the failure had one.
    """

    def __init__(self, kind: str, status_code: int | None = None) -> None:
        super().__init__("Answer generation is temporarily unavailable")
        self.kind = kind
        self.status_code = status_code
