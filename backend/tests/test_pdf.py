import re
from pathlib import Path

import pytest
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate

from app.rag.chunking import chunk_document
from app.rag.documents import Document, load_documents
from app.rag.pdf import pdf_to_markdown
from scripts.build_corpus_pdfs import BODY, HEADINGS, render

REPO_ROOT = Path(__file__).resolve().parents[2]


def make_pdf(tmp_path: Path, markdown: str, name: str = "policy") -> Path:
    source = tmp_path / f"{name}.md"
    source.write_text(markdown, encoding="utf-8")
    target = tmp_path / f"{name}.pdf"
    render(source, target)
    return target


def test_headings_and_paragraphs_survive_the_pdf_round_trip(tmp_path: Path) -> None:
    long = " ".join(["A long sentence that wraps across several lines."] * 12)
    markdown = "\n\n".join(
        [
            "# Leave Policy",
            "Intro line.",
            "## 1. Scope",
            "First paragraph.",
            "Second paragraph.",
            "### 1.1 Detail",
            long,
        ]
    )
    assert pdf_to_markdown(make_pdf(tmp_path, markdown)) == markdown + "\n"


def test_heading_at_a_page_bottom_keeps_its_body_on_the_next_page(tmp_path: Path) -> None:
    target = tmp_path / "split.pdf"
    SimpleDocTemplate(str(target), pagesize=A4).build(
        [
            Paragraph("Title", HEADINGS[1]),
            Paragraph("Body text on the first page.", BODY),
            Paragraph("2. Scope", HEADINGS[2]),
            PageBreak(),
            Paragraph("Body text on the second page.", BODY),
        ]
    )
    assert pdf_to_markdown(target) == (
        "# Title\n\nBody text on the first page.\n\n## 2. Scope\n\nBody text on the second page.\n"
    )


def test_a_paragraph_that_flows_onto_the_next_page_stays_whole(tmp_path: Path) -> None:
    long = " ".join(f"word{i}" for i in range(2500))
    markdown = f"# Title\n\n{long}"
    target = make_pdf(tmp_path, markdown)
    assert pdf_to_markdown(target) == markdown + "\n"


def test_a_pdf_without_a_text_layer_is_rejected(tmp_path: Path) -> None:
    drawing = canvas.Canvas(str(tmp_path / "scan.pdf"), pagesize=A4)
    drawing.rect(100, 100, 200, 200, fill=1)  # an image-like page with no text
    drawing.save()
    assert pdf_to_markdown(tmp_path / "scan.pdf") == ""
    with pytest.raises(ValueError, match="No text layer"):
        load_documents(tmp_path)


def test_an_unreadable_pdf_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "broken.pdf").write_bytes(b"not a pdf")
    with pytest.raises(ValueError, match="Could not read PDF broken.pdf"):
        load_documents(tmp_path)


def test_loader_reads_pdfs_with_their_title(tmp_path: Path) -> None:
    make_pdf(tmp_path, "# Remote Work Policy\n\n## 1. Scope\n\n" + "Body text. " * 20)
    (tmp_path / "policy.md").unlink()  # keep only the PDF, as in the corpus
    [document] = load_documents(tmp_path)
    assert (document.doc_id, document.title) == ("policy", "Remote Work Policy")
    assert [chunk.section for chunk in chunk_document(document)] == ["1. Scope"]


def test_corpus_pdfs_match_their_markdown_sources() -> None:
    sources = sorted((REPO_ROOT / "data" / "source").glob("*.md"))
    pdfs = sorted((REPO_ROOT / "data" / "corpus").glob("*.pdf"))
    assert [path.stem for path in sources] == [path.stem for path in pdfs]
    for source, pdf in zip(sources, pdfs, strict=True):
        text = source.read_text(encoding="utf-8")
        heading = re.search(r"^# (.+)$", text, re.MULTILINE)
        assert heading
        title = heading.group(1)
        from_markdown = chunk_document(Document(doc_id=source.stem, title=title, text=text))
        from_pdf = chunk_document(
            Document(doc_id=source.stem, title=title, text=pdf_to_markdown(pdf))
        )
        assert [(c.section, c.text) for c in from_pdf] == [
            (c.section, c.text) for c in from_markdown
        ], source.name
