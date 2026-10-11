"""Align grouped source markers with the API citation list."""

import re

from app.rag.prompts import NOT_FOUND
from app.rag.retrieval import RetrievedChunk
from app.schemas import Citation, RagAnswer

# A bracket holding only digits, commas, whitespace and dashes, e.g. "[1]", "[1, 3]" or
# "[2-4]". Numeric prose in brackets also matches; see "Citation brackets" in ADR 0002.
CITATION_GROUP = re.compile(r"\[([0-9,\s–-]+)\]")
# One reference inside a group: a number, or a hyphen/en-dash range of numbers.
REFERENCE = re.compile(r"([0-9]+)(?:\s*[-–]\s*([0-9]+))?")
# Models sometimes cite with CJK lenticular or fullwidth brackets, e.g. "【1】" or "［1］".
_BRACKET_VARIANTS = str.maketrans({"【": "[", "】": "]", "［": "[", "］": "]"})


def normalize_brackets(text: str) -> str:
    """Rewrite lenticular and fullwidth brackets as ASCII, so "【1】" reads as "[1]"."""
    return text.translate(_BRACKET_VARIANTS)


def is_not_found(text: str) -> bool:
    """Whether ``text`` is the fixed not-found reply, ignoring any citation markers."""
    return CITATION_GROUP.sub("", normalize_brackets(text)).strip() == NOT_FOUND


def _bounded_number(text: str, limit: int) -> int:
    # Clamp before expanding ranges, including extremely long numeric strings.
    digits = text.lstrip("0") or "0"
    if len(digits) > len(str(limit)):
        return limit + 1
    return min(int(digits), limit + 1)


def _group_numbers(text: str, limit: int) -> list[int]:
    """Expand one group's references into source numbers within ``1..limit``, in order."""
    numbers: list[int] = []
    for match in REFERENCE.finditer(text):
        start = _bounded_number(match.group(1), limit)
        end_text = match.group(2)
        if end_text is None:
            if 1 <= start <= limit:
                numbers.append(start)
            continue
        end = _bounded_number(end_text, limit)
        if start <= end:
            numbers.extend(range(max(1, start), min(limit, end) + 1))
        else:
            numbers.extend(range(min(limit, start), max(1, end) - 1, -1))
    return numbers


def process_citations(text: str, sources: list[RetrievedChunk]) -> RagAnswer:
    """Validate and renumber the citation markers in a generated answer.

    Citation groups are expanded, references outside ``1..len(sources)`` are dropped, and
    the remaining sources are renumbered by first appearance, so the answer's markers
    match the order of the returned citation list.

    Args:
        text: Model output containing ``[n]`` markers (lenticular or fullwidth brackets
            are accepted too).
        sources: The numbered sources the model was shown, in order.

    Returns:
        The rewritten answer and its citations; or the not-found answer with no citations
        if the text is the not-found reply or no valid citation remains.
    """
    text = normalize_brackets(text).strip()
    if is_not_found(text):
        return RagAnswer(answer=NOT_FOUND, kind="not_found")
    # Original source number -> new number, in order of first appearance.
    numbers: dict[int, int] = {}

    def replace(match: re.Match[str]) -> str:
        """Rewrite one bracket group with renumbered references, or remove it."""
        group = []
        for number in _group_numbers(match.group(1), len(sources)):
            if number not in numbers:
                numbers[number] = len(numbers) + 1
            if numbers[number] not in group:
                group.append(numbers[number])
        return "[" + ", ".join(map(str, group)) + "]" if group else ""

    answer = CITATION_GROUP.sub(replace, text).strip()
    # An answer without a single valid citation is ungrounded, so it becomes a refusal.
    if not numbers:
        return RagAnswer(answer=NOT_FOUND, kind="not_found")
    citations = [
        Citation(
            doc_id=sources[number - 1].chunk.doc_id,
            title=sources[number - 1].chunk.title,
            section=sources[number - 1].chunk.section,
            snippet=sources[number - 1].chunk.text,
            score=sources[number - 1].dense_score,
        )
        for number in numbers
    ]
    return RagAnswer(answer=answer, citations=citations)
