from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Tuple


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    source_path: str
    title: str
    heading: str
    text: str


_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)\s*$", re.MULTILINE)


def _split_md_by_headings(text: str) -> List[Tuple[str, str]]:
    """Return list of (heading, section_text) preserving order."""
    matches = list(_HEADING_RE.finditer(text))
    if not matches:
        return [("", text)]
    sections: List[Tuple[str, str]] = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        heading = m.group(2).strip()
        section_text = text[start:end].strip()
        sections.append((heading, section_text))
    return sections


def _chunk_by_paragraphs(text: str, max_chars: int) -> List[str]:
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    out: List[str] = []
    buf: List[str] = []
    cur = 0
    for p in paras:
        if cur + len(p) + 2 > max_chars and buf:
            out.append("\n\n".join(buf))
            buf = [p]
            cur = len(p)
        else:
            buf.append(p)
            cur += len(p) + 2
    if buf:
        out.append("\n\n".join(buf))
    return out


def chunk_document(
    *,
    doc_id: str,
    source_path: str,
    title: str,
    text: str,
    max_chars: int = 2200,
) -> List[Chunk]:
    """Chunk a document into smaller pieces for retrieval."""
    # Markdown-aware chunking
    sections = _split_md_by_headings(text)
    chunks: List[Chunk] = []
    idx = 0
    for heading, sec_text in sections:
        parts = _chunk_by_paragraphs(sec_text, max_chars=max_chars)
        for part in parts:
            idx += 1
            chunk_id = f"{doc_id}__{idx:04d}"
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    source_path=source_path,
                    title=title,
                    heading=heading,
                    text=part.strip(),
                )
            )
    return chunks
