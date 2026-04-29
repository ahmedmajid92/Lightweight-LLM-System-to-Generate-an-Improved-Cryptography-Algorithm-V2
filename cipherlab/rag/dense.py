from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore


@dataclass
class DenseIndex:
    ids: List[str]
    vectors: np.ndarray  # shape (N, D), float32, L2-normalized
    metadata: Dict[str, Any] = field(default_factory=dict)

    def search(self, query_vec: np.ndarray, top_k: int = 6) -> List[Tuple[str, float]]:
        if query_vec.ndim != 1:
            query_vec = query_vec.reshape(-1)
        # cosine similarity since vectors are normalized
        sims = self.vectors @ query_vec
        # top-k indices
        if top_k >= len(self.ids):
            idxs = np.argsort(-sims)
        else:
            idxs = np.argpartition(-sims, top_k)[:top_k]
            idxs = idxs[np.argsort(-sims[idxs])]
        return [(self.ids[i], float(sims[i])) for i in idxs]

    def save(self, ids_path: str, vecs_path: str, metadata_path: Optional[str] = None) -> None:
        with open(ids_path, "w", encoding="utf-8") as f:
            json.dump(self.ids, f, indent=2)
        np.save(vecs_path, self.vectors)
        if metadata_path:
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(self.metadata, f, indent=2)

    @staticmethod
    def load(ids_path: str, vecs_path: str, metadata_path: Optional[str] = None) -> "DenseIndex":
        ids = json.loads(open(ids_path, "r", encoding="utf-8").read())
        vecs = np.load(vecs_path)
        metadata: Dict[str, Any] = {}
        if metadata_path:
            try:
                metadata = json.loads(open(metadata_path, "r", encoding="utf-8").read())
            except FileNotFoundError:
                metadata = {}
        return DenseIndex(ids=ids, vectors=vecs, metadata=metadata)


def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
    return mat / norms


def embed_texts_openai(
    *,
    api_key: str,
    model: str,
    texts: List[str],
    batch_size: int = 64,
) -> np.ndarray:
    if OpenAI is None:
        raise RuntimeError("openai package not installed")
    client = OpenAI(api_key=api_key)
    all_vecs: List[List[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        resp = client.embeddings.create(model=model, input=batch)
        # OpenAI returns embeddings in the same order
        for item in resp.data:
            all_vecs.append(item.embedding)
    mat = np.array(all_vecs, dtype=np.float32)
    return _l2_normalize(mat)


def embed_query_openai(*, api_key: str, model: str, text: str) -> np.ndarray:
    if OpenAI is None:
        raise RuntimeError("openai package not installed")
    client = OpenAI(api_key=api_key)
    resp = client.embeddings.create(model=model, input=text)
    vec = np.array(resp.data[0].embedding, dtype=np.float32)
    # normalize
    vec = vec / (np.linalg.norm(vec) + 1e-12)
    return vec


def _apply_embedding_dim(mat: np.ndarray, dimensions: Optional[int]) -> np.ndarray:
    if dimensions is None:
        return mat
    if mat.shape[1] < dimensions:
        raise ValueError(f"Embedding dimension {mat.shape[1]} is smaller than requested {dimensions}")
    return mat[:, :dimensions]


def embed_texts_local(
    *,
    base_url: str,
    api_key: str,
    model: str,
    texts: List[str],
    dimensions: Optional[int] = None,
    prefix: str = "",
    timeout_seconds: float = 60.0,
    batch_size: int = 32,
) -> np.ndarray:
    """Embed texts through a local llama.cpp OpenAI-compatible embeddings endpoint."""
    if OpenAI is None:
        raise RuntimeError("openai package not installed")
    client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout_seconds)
    all_vecs: List[List[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = [prefix + text for text in texts[i : i + batch_size]]
        resp = client.embeddings.create(model=model, input=batch, encoding_format="float")
        for item in resp.data:
            all_vecs.append(item.embedding)
    mat = np.array(all_vecs, dtype=np.float32)
    mat = _apply_embedding_dim(mat, dimensions)
    return _l2_normalize(mat)


def embed_query_local(
    *,
    base_url: str,
    api_key: str,
    model: str,
    text: str,
    dimensions: Optional[int] = None,
    prefix: str = "",
    timeout_seconds: float = 60.0,
) -> np.ndarray:
    mat = embed_texts_local(
        base_url=base_url,
        api_key=api_key,
        model=model,
        texts=[text],
        dimensions=dimensions,
        prefix=prefix,
        timeout_seconds=timeout_seconds,
        batch_size=1,
    )
    return mat[0]
