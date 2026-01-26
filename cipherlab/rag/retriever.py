from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..config import Settings
from .dense import embed_query_openai
from .hybrid import HybridRetriever
from .index import load_bm25, load_chunks, load_dense


@dataclass
class RetrievedChunk:
    chunk_id: str
    score: float
    source_path: str
    title: str
    heading: str
    text: str
    debug: Dict[str, float]


class RAGRetriever:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._chunks = {row["chunk_id"]: row for row in load_chunks(settings)}
        bm25 = load_bm25(settings)
        dense = load_dense(settings)
        self._hybrid = HybridRetriever(bm25=bm25, dense=dense, alpha=settings.rag_hybrid_alpha, top_k=settings.rag_top_k)

    def retrieve(self, query: str) -> List[RetrievedChunk]:
        query_vec: Optional[np.ndarray] = None
        if self._hybrid.dense is not None:
            if not self.settings.openai_api_key:
                raise RuntimeError("Embeddings index exists but OPENAI_API_KEY missing (needed for query embedding)")
            query_vec = embed_query_openai(
                api_key=self.settings.openai_api_key,
                model=self.settings.openai_embedding_model,
                text=query,
            )
        ranked = self._hybrid.search(query=query, query_vec=query_vec, top_k=self.settings.rag_top_k)
        out: List[RetrievedChunk] = []
        for chunk_id, score, dbg in ranked:
            row = self._chunks.get(chunk_id)
            if not row:
                continue
            out.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    score=float(score),
                    source_path=row.get("source_path", ""),
                    title=row.get("title", ""),
                    heading=row.get("heading", ""),
                    text=row.get("text", ""),
                    debug={k: float(v) for k, v in dbg.items()},
                )
            )
        return out

    @staticmethod
    def format_for_prompt(chunks: List[RetrievedChunk], *, max_chars: int = 9000) -> str:
        parts: List[str] = []
        total = 0
        for i, c in enumerate(chunks, start=1):
            header = f"[KB {i}] {c.title} :: {c.heading} ({c.source_path})"
            body = c.text.strip()
            block = header + "\n" + body
            if total + len(block) > max_chars:
                break
            parts.append(block)
            total += len(block)
        return "\n\n".join(parts)
