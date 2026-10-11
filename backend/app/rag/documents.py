"""Read the small, local PDF, Markdown and text corpus deterministically."""

import re
from pathlib import Path

from pydantic import BaseModel

from app.rag.pdf import pdf_to_markdown

SUPPORTED_SUFFIXES = {".pdf", ".md", ".txt"}


class Document(BaseModel):
    """A corpus file: ``doc_id`` is its file stem; ``text`` is Markdown or plain text."""

    doc_id: str
    title: str
    text: str


def load_documents(directory: Path) -> list[Document]:
    """Load every PDF, Markdown and text file directly inside ``directory``, by name.

    PDFs are converted to Markdown first (:func:`~app.rag.pdf.pdf_to_markdown`), so all
    formats share one chunker. The title is the first line starting with ``# ``, falling
    back to the file stem. This rule is separate from the chunker's heading parser; see
    "Title parsing" under known issues in decision record 0002.

    Raises:
        ValueError: If the directory doesn't exist, two files share a stem, or a PDF
            can't be read or has no text layer (for example, a scanned document).
    """
    if not directory.is_dir():
        raise ValueError(f"Corpus directory does not exist: {directory}")
    documents = []
    seen = set()
    for file in sorted(directory.iterdir()):
        if not file.is_file() or file.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        if file.stem in seen:
            raise ValueError(f"Duplicate corpus doc_id: {file.stem}")
        seen.add(file.stem)
        if file.suffix.lower() == ".pdf":
            try:
                text = pdf_to_markdown(file)
            except Exception as error:
                raise ValueError(f"Could not read PDF {file.name}") from error
            # Fail loudly rather than silently index nothing for this document.
            if not text.strip():
                raise ValueError(f"No text layer in {file.name}; scanned PDFs need OCR")
        else:
            # utf-8-sig drops the byte-order mark some Windows editors add.
            text = file.read_text(encoding="utf-8-sig")
        title = re.search(r"^# (.+)$", text, re.MULTILINE)
        documents.append(
            Document(
                doc_id=file.stem, title=title.group(1).strip() if title else file.stem, text=text
            )
        )
    return documents
