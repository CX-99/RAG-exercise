"""Convert text-based PDFs to Markdown so the section-aware chunker can process them.

Converting to Markdown first is the usual RAG ingestion pattern: the chunker, section
paths and citations then work the same for every source format.
"""

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pdfplumber

# A line at least this much larger than body text is a heading; smaller differences
# are treated as body text, so slight size variation does not create headings.
HEADING_SIZE_RATIO = 1.15
# A gap this much larger than the usual line spacing starts a new paragraph.
PARAGRAPH_GAP_RATIO = 1.3


@dataclass
class _Line:
    page: int
    top: float
    size: float
    text: str


def _read_lines(path: Path) -> list[_Line]:
    """Extract text lines with their position and dominant font size, page by page."""
    lines: list[_Line] = []
    with pdfplumber.open(path) as pdf:
        for number, page in enumerate(pdf.pages, start=1):
            for line in page.extract_text_lines(return_chars=True):
                text = line["text"].strip()
                if not text:
                    continue
                sizes = Counter(round(char["size"], 1) for char in line["chars"])
                lines.append(_Line(number, line["top"], sizes.most_common(1)[0][0], text))
    return lines


def pdf_to_markdown(path: Path) -> str:
    """Return the PDF's text as Markdown, with headings recovered from font size.

    Body size is the size covering the most characters, which assumes body text makes
    up most of the document, as it does in policies. Each larger size, at least
    :data:`HEADING_SIZE_RATIO` times the body size, becomes a heading level, largest
    first; at most six levels are used, as in Markdown. Body lines are joined into
    paragraphs, and a vertical gap well above the usual line spacing starts a new
    paragraph. A paragraph that continues onto the next page stays one paragraph.

    Limitations: scanned PDFs (no text layer), tables and multi-column layouts are
    not handled, and headings are recognized only by size, not by bold weight.
    """
    lines = _read_lines(path)
    if not lines:
        return ""
    weight = Counter()
    for line in lines:
        weight[line.size] += len(line.text)
    body_size = weight.most_common(1)[0][0]
    heading_sizes = sorted(
        {line.size for line in lines if line.size >= body_size * HEADING_SIZE_RATIO},
        reverse=True,
    )[:6]
    levels = {size: level for level, size in enumerate(heading_sizes, start=1)}
    gaps = sorted(
        later.top - earlier.top
        for earlier, later in zip(lines, lines[1:], strict=False)
        if earlier.page == later.page
        and earlier.size == later.size == body_size
        and later.top > earlier.top
    )
    # The lower quartile is the line spacing inside paragraphs, assuming most body lines
    # belong to multi-line paragraphs; the larger gaps between paragraphs sit above it.
    usual_gap = gaps[len(gaps) // 4] if gaps else body_size * 1.2

    blocks: list[str] = []
    paragraph: list[str] = []
    previous: _Line | None = None
    for line in lines:
        level = levels.get(line.size)
        if level is not None:
            if paragraph:
                blocks.append(" ".join(paragraph))
                paragraph = []
            heading = "#" * level + " "
            # A heading that wraps onto a second line continues the same heading.
            if previous is not None and levels.get(previous.size) == level and blocks:
                blocks[-1] = f"{blocks[-1]} {line.text}"
            else:
                blocks.append(heading + line.text)
        else:
            new_paragraph = (
                previous is not None
                and previous.page == line.page
                and levels.get(previous.size) is None
                and line.top - previous.top > usual_gap * PARAGRAPH_GAP_RATIO
            )
            if new_paragraph and paragraph:
                blocks.append(" ".join(paragraph))
                paragraph = []
            paragraph.append(line.text)
        previous = line
    if paragraph:
        blocks.append(" ".join(paragraph))
    return "\n\n".join(blocks) + "\n"
