"""Preserve heading ancestry and raw paragraph text for citations."""

import re

from pydantic import BaseModel

from app.rag.documents import Document

# ATX heading. Trailing "#"s are a closing sequence only after whitespace, so "## C#" keeps "C#".
HEADING = re.compile(r"^(#{1,6})[ \t]+(.+?)(?:[ \t]+#+)?[ \t]*$")


class Chunk(BaseModel):
    """A retrievable unit: one section, or a paragraph-aligned part of a long section."""

    chunk_id: str
    doc_id: str
    title: str
    section: str
    text: str

    @property
    def indexed_text(self) -> str:
        """Text used for embedding and BM25: ``title — section``, a blank line, then the body.

        The prefix makes each chunk self-describing for retrieval; citations show the raw
        ``text`` instead.
        """
        return f"{self.title} — {self.section}\n\n{self.text}"


def split_paragraphs(text: str, max_chars: int) -> list[str]:
    """Pack consecutive paragraphs into parts of at most ``max_chars`` characters.

    The limit is soft: a single paragraph longer than ``max_chars`` is kept whole rather
    than cut mid-sentence.
    """
    parts: list[str] = []
    current = ""
    for paragraph in re.split(r"\n\s*\n", text.strip()):
        if not paragraph.strip():
            continue
        candidate = f"{current}\n\n{paragraph}" if current else paragraph
        if current and len(candidate) > max_chars:
            parts.append(current)
            current = paragraph
        else:
            current = candidate
    if current:
        parts.append(current)
    return parts


def chunk_document(document: Document, max_chars: int = 1200) -> list[Chunk]:
    """Split a Markdown or plain-text document into section-aware chunks.

    Headings define each chunk's section path. The first H1 is the document title, so it
    is left out of the path; later H1s start new top-level sections. Lines inside code
    fences are never treated as headings. Text before any heading goes in "Introduction".

    Args:
        document: The loaded document.
        max_chars: Soft size limit passed to :func:`split_paragraphs`.

    Returns:
        Chunks in document order, with IDs of the form ``<doc_id>:<n>``.

    Raises:
        ValueError: If ``max_chars`` is less than 1.
    """
    if max_chars < 1:
        raise ValueError("max_chars must be positive")
    chunks: list[Chunk] = []
    headings: list[tuple[int, str]] = []
    lines: list[str] = []
    fence: str | None = None
    seen_title = False

    def flush() -> None:
        """Emit chunks for the lines collected under the current heading path."""
        section = " > ".join(name for _, name in headings) or "Introduction"
        for text in split_paragraphs("\n".join(lines), max_chars):
            chunks.append(
                Chunk(
                    chunk_id=f"{document.doc_id}:{len(chunks)}",
                    doc_id=document.doc_id,
                    title=document.title,
                    section=section,
                    text=text,
                )
            )
        lines.clear()

    for line in document.text.splitlines():
        stripped = line.lstrip()
        if stripped.startswith(("~~~", "```")):
            marker = stripped[:3]
            if fence is None:
                fence = marker
            elif fence == marker:
                fence = None
            lines.append(line)
            continue
        heading = HEADING.match(line) if fence is None else None
        if heading:
            flush()
            level, name = len(heading.group(1)), heading.group(2)
            while headings and headings[-1][0] >= level:
                headings.pop()
            # Only the first H1 is the title; later H1s are top-level sections.
            if level == 1 and not seen_title:
                seen_title = True
            else:
                headings.append((level, name))
        else:
            lines.append(line)
    flush()
    return chunks
