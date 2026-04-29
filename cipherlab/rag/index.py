from __future__ import annotations

import json
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..config import Settings
from .bm25 import BM25Index, build_bm25
from .chunker import Chunk, chunk_document
from .dense import DenseIndex, embed_texts_local, embed_texts_openai
from .documents import load_documents_from_dirs


logger = logging.getLogger("cipherlab.rag")


def _build_dense_vectors(settings: Settings, texts: List[str]) -> Tuple[np.ndarray, Dict[str, Any]]:
    provider = settings.rag_embedding_provider
    errors: List[str] = []

    if provider in {"auto", "local"} and settings.rag_local_embeddings_enabled:
        try:
            vecs = embed_texts_local(
                base_url=settings.rag_local_embedding_base_url,
                api_key=settings.local_llm_api_key,
                model=settings.rag_local_embedding_model,
                texts=texts,
                dimensions=settings.rag_local_embedding_dim,
                prefix="search_document: ",
                timeout_seconds=settings.rag_local_embedding_timeout_seconds,
            )
            metadata = {
                "embedding_provider": "local",
                "embedding_model": settings.rag_local_embedding_model,
                "embedding_dimension": int(vecs.shape[1]),
                "prefix_scheme": "nomic-search-document-query",
                "document_prefix": "search_document: ",
                "query_prefix": "search_query: ",
                "normalization": "l2",
                "dense_weight": settings.rag_hybrid_alpha,
                "bm25_weight": settings.rag_bm25_weight,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            return vecs, metadata
        except Exception as exc:
            errors.append(f"local embeddings failed: {exc}")
            if provider == "local":
                raise
            logger.warning("[rag] Local embedding index build failed: %s. Trying online embeddings.", exc)

    if provider in {"auto", "openai"}:
        if settings.openai_api_key:
            vecs = embed_texts_openai(
                api_key=settings.openai_api_key,
                model=settings.openai_embedding_model,
                texts=texts,
            )
            metadata = {
                "embedding_provider": "openai",
                "embedding_model": settings.openai_embedding_model,
                "embedding_dimension": int(vecs.shape[1]),
                "prefix_scheme": "none",
                "normalization": "l2",
                "dense_weight": settings.rag_hybrid_alpha,
                "bm25_weight": settings.rag_bm25_weight,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            return vecs, metadata
        errors.append("OPENAI_API_KEY is missing")

    raise RuntimeError("Could not build dense embeddings: " + "; ".join(errors))


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
        vecs, metadata = _build_dense_vectors(settings, bm25_texts)
        dense = DenseIndex(ids=[c.chunk_id for c in chunks], vectors=vecs, metadata=metadata)
        dense.save(
            str(out_dir / "dense_ids.json"),
            str(out_dir / "embeddings.npy"),
            str(out_dir / "dense_metadata.json"),
        )


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
    metadata_path = root / settings.kb_index_dir / "dense_metadata.json"
    if not ids_path.exists() or not vecs_path.exists():
        return None
    return DenseIndex.load(str(ids_path), str(vecs_path), str(metadata_path))
