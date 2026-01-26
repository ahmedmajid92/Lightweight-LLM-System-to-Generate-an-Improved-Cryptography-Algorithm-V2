from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from ..config import Settings
from .bm25 import BM25Index, build_bm25
from .chunker import Chunk, chunk_document
from .dense import DenseIndex, embed_texts_openai
from .documents import load_documents_from_dirs


def build_kb_index(settings: Settings) -> None:
    root = Path(settings.project_root)
    kb_dir = root / settings.kb_dir
    ext_dir = root / settings.kb_external_dir
    out_dir = root / settings.kb_index_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    docs = load_documents_from_dirs([kb_dir, ext_dir])
    chunks: List[Chunk] = []
    for doc in docs:
        chunks.extend(
            chunk_document(
                doc_id=doc.doc_id,
                source_path=doc.source_path,
                title=doc.title,
                text=doc.text,
            )
        )

    # Write chunks.jsonl
    chunks_path = out_dir / "chunks.jsonl"
    with chunks_path.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(
                json.dumps(
                    {
                        "chunk_id": c.chunk_id,
                        "doc_id": c.doc_id,
                        "source_path": c.source_path,
                        "title": c.title,
                        "heading": c.heading,
                        "text": c.text,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    # Build BM25 on chunk texts (title+heading+text)
    bm25_texts = [f"{c.title}\n{c.heading}\n{c.text}" for c in chunks]
    bm25 = build_bm25([c.chunk_id for c in chunks], bm25_texts)
    (out_dir / "bm25.json").write_text(json.dumps(bm25.to_json(), indent=2), encoding="utf-8")

    # Optional embeddings
    if settings.rag_use_embeddings:
        if not settings.openai_api_key:
            raise RuntimeError("RAG_USE_EMBEDDINGS=true but OPENAI_API_KEY is missing")
        vecs = embed_texts_openai(
            api_key=settings.openai_api_key,
            model=settings.openai_embedding_model,
            texts=bm25_texts,
        )
        dense = DenseIndex(ids=[c.chunk_id for c in chunks], vectors=vecs)
        dense.save(str(out_dir / "dense_ids.json"), str(out_dir / "embeddings.npy"))


def load_chunks(settings: Settings) -> List[dict]:
    root = Path(settings.project_root)
    chunks_path = root / settings.kb_index_dir / "chunks.jsonl"
    if not chunks_path.exists():
        raise FileNotFoundError("KB index not found. Run: python scripts/build_kb_index.py")
    rows: List[dict] = []
    for line in chunks_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    return rows


def load_bm25(settings: Settings) -> BM25Index:
    root = Path(settings.project_root)
    bm25_path = root / settings.kb_index_dir / "bm25.json"
    obj = json.loads(bm25_path.read_text(encoding="utf-8"))
    return BM25Index.from_json(obj)


def load_dense(settings: Settings) -> Optional[DenseIndex]:
    root = Path(settings.project_root)
    ids_path = root / settings.kb_index_dir / "dense_ids.json"
    vecs_path = root / settings.kb_index_dir / "embeddings.npy"
    if not ids_path.exists() or not vecs_path.exists():
        return None
    return DenseIndex.load(str(ids_path), str(vecs_path))
