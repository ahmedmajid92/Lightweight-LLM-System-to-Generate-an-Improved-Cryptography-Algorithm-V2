from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .bm25 import BM25Index
from .dense import DenseIndex


@dataclass
class HybridRetriever:
    bm25: BM25Index
    dense: Optional[DenseIndex] = None
    alpha: float = 0.55  # weight on dense
    top_k: int = 6

    def search(
        self,
        *,
        query: str,
        query_vec: Optional[np.ndarray] = None,
        top_k: Optional[int] = None,
    ) -> List[Tuple[str, float, Dict[str, float]]]:
        k = top_k or self.top_k

        sparse = self.bm25.score(query, top_k=max(k * 3, k))
        sparse_dict = {doc_id: score for doc_id, score in sparse}
        sparse_max = max(sparse_dict.values(), default=1.0) or 1.0

        dense_list: List[Tuple[str, float]] = []
        if self.dense is not None and query_vec is not None:
            dense_list = self.dense.search(query_vec, top_k=max(k * 3, k))
        dense_dict = {doc_id: score for doc_id, score in dense_list}

        # Normalize and combine
        combined: Dict[str, float] = {}
        details: Dict[str, Dict[str, float]] = {}
        ids = set(sparse_dict) | set(dense_dict)
        for doc_id in ids:
            s = sparse_dict.get(doc_id, 0.0) / sparse_max
            d_raw = dense_dict.get(doc_id, 0.0)
            # cosine in [-1,1] -> [0,1]
            d = (d_raw + 1.0) / 2.0
            score = (1.0 - self.alpha) * s + self.alpha * d
            combined[doc_id] = score
            details[doc_id] = {"sparse_norm": s, "dense_cos": d_raw, "dense_norm": d}

        ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)[:k]
        return [(doc_id, score, details[doc_id]) for doc_id, score in ranked]
