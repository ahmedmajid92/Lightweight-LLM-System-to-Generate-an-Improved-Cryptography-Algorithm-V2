from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple


_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def tokenize(text: str) -> List[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


@dataclass
class BM25Index:
    doc_ids: List[str]
    doc_lens: List[int]
    avgdl: float
    idf: Dict[str, float]
    tfs: List[Dict[str, int]]
    k1: float = 1.5
    b: float = 0.75

    def score(self, query: str, *, top_k: int = 6) -> List[Tuple[str, float]]:
        q_toks = tokenize(query)
        if not q_toks:
            return []
        scores = [0.0 for _ in self.doc_ids]
        for term in q_toks:
            if term not in self.idf:
                continue
            idf = self.idf[term]
            for i, tf in enumerate(self.tfs):
                f = tf.get(term, 0)
                if f == 0:
                    continue
                dl = self.doc_lens[i]
                denom = f + self.k1 * (1.0 - self.b + self.b * (dl / self.avgdl))
                scores[i] += idf * (f * (self.k1 + 1.0) / denom)
        pairs = list(zip(self.doc_ids, scores))
        pairs.sort(key=lambda x: x[1], reverse=True)
        return pairs[:top_k]

    def to_json(self) -> dict:
        return {
            "doc_ids": self.doc_ids,
            "doc_lens": self.doc_lens,
            "avgdl": self.avgdl,
            "idf": self.idf,
            "tfs": self.tfs,
            "k1": self.k1,
            "b": self.b,
        }

    @staticmethod
    def from_json(obj: dict) -> "BM25Index":
        return BM25Index(
            doc_ids=list(obj["doc_ids"]),
            doc_lens=list(obj["doc_lens"]),
            avgdl=float(obj["avgdl"]),
            idf={str(k): float(v) for k, v in obj["idf"].items()},
            tfs=[{str(k): int(v) for k, v in tf.items()} for tf in obj["tfs"]],
            k1=float(obj.get("k1", 1.5)),
            b=float(obj.get("b", 0.75)),
        )


def build_bm25(doc_ids: List[str], texts: List[str]) -> BM25Index:
    tfs: List[Dict[str, int]] = []
    doc_lens: List[int] = []
    df: Dict[str, int] = {}

    for text in texts:
        toks = tokenize(text)
        doc_lens.append(len(toks))
        tf: Dict[str, int] = {}
        for t in toks:
            tf[t] = tf.get(t, 0) + 1
        tfs.append(tf)
        for t in set(toks):
            df[t] = df.get(t, 0) + 1

    N = len(texts)
    avgdl = sum(doc_lens) / N if N else 1.0

    # Okapi BM25 IDF
    idf: Dict[str, float] = {}
    for term, n_q in df.items():
        # adding 0.5 to avoid negative idf for extremely frequent terms
        idf[term] = math.log(1 + (N - n_q + 0.5) / (n_q + 0.5))

    return BM25Index(doc_ids=doc_ids, doc_lens=doc_lens, avgdl=avgdl, idf=idf, tfs=tfs)
