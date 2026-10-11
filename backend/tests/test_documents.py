from pathlib import Path

import pytest

from app.rag.chunking import chunk_document
from app.rag.documents import Document, load_documents


def test_heading_paths_and_raw_snippets() -> None:
    document = Document(
        doc_id="leave",
        title="Leave",
        text=(
            "# Leave\nPreface.\n## 3. Annual Leave\nParent.\n"
            "### 3.2 Carry-over\nKeep five days.\n### 3.3 Requests\nAsk first.\n"
            "## 4. Records\nKeep records."
        ),
    )
    chunks = chunk_document(document)
    assert [chunk.section for chunk in chunks] == [
        "Introduction",
        "3. Annual Leave",
        "3. Annual Leave > 3.2 Carry-over",
        "3. Annual Leave > 3.3 Requests",
        "4. Records",
    ]
    assert chunks[2].text == "Keep five days."
    assert chunks[2].indexed_text == "Leave — 3. Annual Leave > 3.2 Carry-over\n\nKeep five days."
    assert len({chunk.chunk_id for chunk in chunks}) == 5


def test_long_sections_split_at_paragraph_boundaries_without_loss() -> None:
    paragraphs = ["a" * 700, "b" * 700, "c" * 100]
    chunks = chunk_document(
        Document(doc_id="long", title="Long", text="## 1. Rule\n" + "\n\n".join(paragraphs))
    )
    assert [chunk.text for chunk in chunks] == [paragraphs[0], "\n\n".join(paragraphs[1:])]
    assert all(chunk.section == "1. Rule" for chunk in chunks)


def test_single_oversized_paragraph_is_a_documented_soft_limit() -> None:
    text = "word " * 400
    chunks = chunk_document(Document(doc_id="long", title="Long", text=text))
    assert len(chunks) == 1
    assert chunks[0].text == text.strip()


def test_code_fences_do_not_create_sections() -> None:
    text = "## 1. Examples\n~~~text\n## Not a heading\n~~~\nExample."
    chunks = chunk_document(Document(doc_id="code", title="Code", text=text))
    assert len(chunks) == 1
    assert chunks[0].section == "1. Examples"
    assert "## Not a heading" in chunks[0].text


def test_loader_uses_first_title_and_falls_back_to_stem(tmp_path: Path) -> None:
    (tmp_path / "plain.txt").write_text("Plain text.", encoding="utf-8")
    (tmp_path / "headed.md").write_text(
        "# First title\n## Section\nBody.\n# Later", encoding="utf-8"
    )
    (tmp_path / "ignored.docx").write_text("Ignored", encoding="utf-8")
    documents = load_documents(tmp_path)
    assert [(doc.doc_id, doc.title) for doc in documents] == [
        ("headed", "First title"),
        ("plain", "plain"),
    ]
    assert chunk_document(documents[1])[0].section == "Introduction"


def test_duplicate_stems_rejected(tmp_path: Path) -> None:
    for name in ["same.md", "same.txt"]:
        (tmp_path / name).write_text("Text", encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate"):
        load_documents(tmp_path)


def test_missing_corpus_and_blank_document(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="does not exist"):
        load_documents(tmp_path / "missing")
    assert chunk_document(Document(doc_id="blank", title="Blank", text="  ")) == []


@pytest.mark.parametrize(
    ("heading", "expected"),
    [("## C#", "C#"), ("## C# ###", "C#"), ("## Name###", "Name###"), ("## Name ###", "Name")],
)
def test_closing_heading_marks_require_whitespace(heading: str, expected: str) -> None:
    chunks = chunk_document(Document(doc_id="code", title="Code", text=heading + "\nBody."))
    assert chunks[0].section == expected


def test_later_h1_is_a_section_with_its_own_descendants() -> None:
    chunks = chunk_document(
        Document(
            doc_id="multi",
            title="Title",
            text=(
                "# Title\nIntro.\n## First\nOne.\n"
                "# Second part\nTwo.\n## Child\nThree.\n# Third part\nFour."
            ),
        )
    )
    assert [chunk.section for chunk in chunks] == [
        "Introduction",
        "First",
        "Second part",
        "Second part > Child",
        "Third part",
    ]
    assert [chunk.text for chunk in chunks] == ["Intro.", "One.", "Two.", "Three.", "Four."]
