from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

from pypdf import PdfReader


@dataclass(frozen=True)
class Document:
    doc_id: str
    source_path: str
    title: str
    text: str


def _read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _read_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    parts: List[str] = []
    for page in reader.pages:
        try:
            parts.append(page.extract_text() or "")
        except Exception:
            continue
    return "\n".join(parts)


def load_documents_from_dirs(dirs: List[str | Path]) -> List[Document]:
    """Load .md/.txt/.pdf files recursively from given directories."""
    out: List[Document] = []
    for d in dirs:
        root = Path(d)
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix.lower() not in {".md", ".txt", ".pdf"}:
                continue
            rel = str(path)
            title = path.stem
            if path.suffix.lower() == ".pdf":
                text = _read_pdf(path)
            else:
                text = _read_text_file(path)
            text = text.strip()
            if not text:
                continue
            doc_id = re.sub(r"[^a-zA-Z0-9_]+", "_", rel)[-180:]
            out.append(Document(doc_id=doc_id, source_path=rel, title=title, text=text))
    return out
