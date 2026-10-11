"""Render the Markdown policy sources in data/source/ as the PDF corpus in data/corpus/.

The policies are fictional, so their PDFs are generated rather than collected. Rendering
uses reportlab's invariant mode, so rerunning the script produces byte-identical files.

Usage, from backend/:
    python -m uv run python scripts/build_corpus_pdfs.py
"""

import re
import sys
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab import rl_config
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = REPO_ROOT / "data" / "source"
CORPUS_DIR = REPO_ROOT / "data" / "corpus"

BODY = ParagraphStyle("body", fontName="Helvetica", fontSize=10, leading=13, spaceAfter=6)
# Headings differ from body text by size, which is what the PDF reader detects.
HEADINGS = {
    1: ParagraphStyle("h1", fontName="Helvetica-Bold", fontSize=20, leading=24, spaceAfter=10),
    2: ParagraphStyle(
        "h2", fontName="Helvetica-Bold", fontSize=15, leading=19, spaceBefore=8, spaceAfter=6
    ),
    3: ParagraphStyle(
        "h3", fontName="Helvetica-Bold", fontSize=12.5, leading=16, spaceBefore=6, spaceAfter=4
    ),
}
HEADING = re.compile(r"^(#{1,3}) (.+)$")


def blocks(markdown: str) -> list[tuple[ParagraphStyle, str]]:
    """Split Markdown into styled blocks: one per heading line or paragraph."""
    result: list[tuple[ParagraphStyle, str]] = []
    for chunk in re.split(r"\n\s*\n", markdown.strip()):
        paragraph: list[str] = []
        for line in chunk.splitlines():
            heading = HEADING.match(line)
            if heading:
                if paragraph:
                    result.append((BODY, " ".join(paragraph)))
                    paragraph = []
                result.append((HEADINGS[len(heading.group(1))], heading.group(2)))
            elif line.strip():
                paragraph.append(line.strip())
        if paragraph:
            result.append((BODY, " ".join(paragraph)))
    return result


def render(source: Path, target: Path) -> None:
    """Render one Markdown policy as an A4 PDF."""
    styled = blocks(source.read_text(encoding="utf-8"))
    title = next((text for style, text in styled if style is HEADINGS[1]), source.stem)
    document = SimpleDocTemplate(
        str(target),
        pagesize=A4,
        leftMargin=22 * mm,
        rightMargin=22 * mm,
        topMargin=20 * mm,
        bottomMargin=20 * mm,
        title=title,
        author="AI chatbot PoC (fictional sample)",
        subject="Fictional sample policy for the Policy Assistant PoC",
    )
    document.build([Paragraph(escape(text), style) for style, text in styled])


def main() -> int:
    """Render every source file; return a process exit code."""
    rl_config.invariant = 1  # Fixed timestamps and IDs, for reproducible files.
    sources = sorted(SOURCE_DIR.glob("*.md"))
    if not sources:
        print(f"No Markdown sources in {SOURCE_DIR}", file=sys.stderr)
        return 1
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    for source in sources:
        target = CORPUS_DIR / f"{source.stem}.pdf"
        render(source, target)
        print(f"{source.name} -> {target.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
