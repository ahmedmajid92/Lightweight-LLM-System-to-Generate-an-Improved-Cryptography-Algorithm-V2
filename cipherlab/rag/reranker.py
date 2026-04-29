from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import List, Protocol

try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore


logger = logging.getLogger("cipherlab.rag")


class RerankableChunk(Protocol):
    chunk_id: str
    score: float
    title: str
    heading: str
    text: str


@dataclass
class RerankScore:
    chunk_id: str
    score: float


def _clean_json_text(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text.strip(), flags=re.DOTALL).strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end >= start:
        return text[start : end + 1]
    return text


class LocalChatReranker:
    """Optional local reranker using an OpenAI-compatible llama.cpp server.

    The reranker is intentionally off by default. It is designed for small
    candidate sets after hybrid retrieval, not as a first-stage retriever.
    """

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout_seconds: float = 60.0,
    ):
        if OpenAI is None:
            raise RuntimeError("openai package not installed")
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout_seconds)
        self.model = model

    def rerank(
        self,
        *,
        query: str,
        chunks: List[RerankableChunk],
        top_k: int,
    ) -> List[RerankableChunk]:
        if not chunks:
            return []

        candidates = [
            {
                "chunk_id": c.chunk_id,
                "title": c.title,
                "heading": c.heading,
                "text": c.text[:1200],
            }
            for c in chunks
        ]
        system = (
            "You are a retrieval reranker for a lightweight cryptography research system. "
            "Score each candidate for how directly it helps answer the query. "
            "Prefer exact component IDs, algorithm names, architecture constraints, and metric terminology. "
            "Return ONLY JSON with this shape: {\"scores\": [{\"chunk_id\": \"...\", \"score\": 0.0}]}. "
            "Scores must be between 0 and 1."
        )
        user = "Query:\n" + query + "\n\nCandidates:\n" + json.dumps(candidates, ensure_ascii=False)

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.0,
            max_tokens=2048,
        )
        text = _clean_json_text(resp.choices[0].message.content or "")
        try:
            payload = json.loads(text)
            scores = payload.get("scores", [])
            score_map = {
                str(item.get("chunk_id")): float(item.get("score", 0.0))
                for item in scores
                if isinstance(item, dict) and item.get("chunk_id")
            }
        except Exception as exc:
            logger.warning("[rag] Local reranker returned invalid JSON: %s. Keeping hybrid order.", exc)
            return chunks[:top_k]

        reranked = sorted(
            chunks,
            key=lambda c: (score_map.get(c.chunk_id, -1.0), c.score),
            reverse=True,
        )
        return reranked[:top_k]
