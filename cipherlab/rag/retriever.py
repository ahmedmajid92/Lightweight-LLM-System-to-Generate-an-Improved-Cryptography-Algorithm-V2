from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..config import Settings
from .dense import embed_query_local, embed_query_openai
from .hybrid import HybridRetriever
from .index import load_bm25, load_chunks, load_dense
from .reranker import LocalChatReranker


logger = logging.getLogger("cipherlab.rag")


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

    def _embed_query_for_dense_index(self, query: str) -> Optional[np.ndarray]:
        dense = self._hybrid.dense
        if dense is None:
            return None
        metadata = dense.metadata or {}
        provider = str(metadata.get("embedding_provider") or "openai").lower()
        try:
            if provider == "local":
                vec = embed_query_local(
                    base_url=self.settings.rag_local_embedding_base_url,
                    api_key=self.settings.local_llm_api_key,
                    model=str(metadata.get("embedding_model") or self.settings.rag_local_embedding_model),
                    text=query,
                    dimensions=int(metadata.get("embedding_dimension") or self.settings.rag_local_embedding_dim),
                    prefix=str(metadata.get("query_prefix") or "search_query: "),
                    timeout_seconds=self.settings.rag_local_embedding_timeout_seconds,
                )
            else:
                if not self.settings.openai_api_key:
                    logger.warning("[rag] Dense index exists but online embedding key is missing; using BM25 only.")
                    return None
                vec = embed_query_openai(
                    api_key=self.settings.openai_api_key,
                    model=str(metadata.get("embedding_model") or self.settings.openai_embedding_model),
                    text=query,
                )
        except Exception as exc:
            logger.warning("[rag] Dense query embedding failed: %s. Using BM25 only.", exc)
            return None

        expected_dim = int(dense.vectors.shape[1])
        if int(vec.shape[0]) != expected_dim:
            logger.warning(
                "[rag] Dense query dimension %s does not match index dimension %s; using BM25 only.",
                int(vec.shape[0]),
                expected_dim,
            )
            return None
        return vec

    def retrieve(self, query: str) -> List[RetrievedChunk]:
        query_vec = self._embed_query_for_dense_index(query)
        final_k = self.settings.rag_final_top_k
        candidate_k = final_k
        if self.settings.rag_rerank_enabled:
            candidate_k = max(self.settings.rag_rerank_top_n, final_k)
        ranked = self._hybrid.search(query=query, query_vec=query_vec, top_k=candidate_k)
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
        if self.settings.rag_rerank_enabled and out:
            try:
                reranker = LocalChatReranker(
                    base_url=self.settings.rag_rerank_base_url,
                    api_key=self.settings.local_llm_api_key,
                    model=self.settings.rag_rerank_model,
                    timeout_seconds=self.settings.local_llm_timeout_seconds,
                )
                return reranker.rerank(query=query, chunks=out, top_k=final_k)
            except Exception as exc:
                logger.warning("[rag] Local reranker failed: %s. Keeping hybrid order.", exc)
        return out[:final_k]

    @staticmethod
    def format_for_prompt(chunks: List[RetrievedChunk], *, max_chars: int = 9000) -> str:
        if not chunks:
            return ""
        guidance = (
            "RAG grounding instructions:\n"
            "- Use these chunks as supporting evidence; do not invent component IDs or algorithm claims.\n"
            "- Preserve exact identifiers such as sbox.aes, perm.present, linear.aes_mixcolumns, and ks.sha256_kdf.\n"
            "- Treat security conclusions as hypotheses unless validated by deterministic metrics.\n"
        )
        parts: List[str] = [guidance]
        total = len(guidance)
        for i, c in enumerate(chunks, start=1):
            header = f"[KB {i}] {c.title} :: {c.heading} ({c.source_path})"
            body = c.text.strip()
            block = header + "\n" + body
            if total + len(block) > max_chars:
                break
            parts.append(block)
            total += len(block)
        return "\n\n".join(parts)
