from types import SimpleNamespace

import numpy as np

from cipherlab.config import Settings
from cipherlab.rag import dense as dense_module
from cipherlab.rag.dense import DenseIndex, embed_texts_local
from cipherlab.rag.retriever import RAGRetriever


def test_local_embeddings_apply_nomic_prefix_and_dimension(monkeypatch):
    calls = []

    class FakeEmbeddings:
        def create(self, **kwargs):
            calls.append(kwargs)
            data = []
            for _ in kwargs["input"]:
                data.append(SimpleNamespace(embedding=[1.0] * 768))
            return SimpleNamespace(data=data)

    class FakeOpenAI:
        def __init__(self, **kwargs):
            self.embeddings = FakeEmbeddings()

    monkeypatch.setattr(dense_module, "OpenAI", FakeOpenAI)

    vecs = embed_texts_local(
        base_url="http://127.0.0.1:8083/v1",
        api_key="sk-no-key-required",
        model="nomic-embed-text-v1.5",
        texts=["SPN S-box diffusion"],
        dimensions=512,
        prefix="search_document: ",
    )

    assert calls[0]["input"] == ["search_document: SPN S-box diffusion"]
    assert vecs.shape == (1, 512)
    assert np.isclose(np.linalg.norm(vecs[0]), 1.0)


def test_rag_retriever_uses_bm25_when_dense_query_dimension_mismatches(monkeypatch):
    chunks = [
        {
            "chunk_id": "c1",
            "source_path": "kb/test.md",
            "title": "SPN",
            "heading": "S-boxes",
            "text": "Use sbox.aes for confusion.",
        }
    ]
    dense = DenseIndex(
        ids=["c1"],
        vectors=np.ones((1, 512), dtype=np.float32),
        metadata={
            "embedding_provider": "local",
            "embedding_model": "nomic-embed-text-v1.5",
            "embedding_dimension": 512,
            "query_prefix": "search_query: ",
        },
    )

    class FakeBM25:
        def score(self, query, *, top_k=6):
            return [("c1", 2.0)]

    monkeypatch.setattr("cipherlab.rag.retriever.load_chunks", lambda settings: chunks)
    monkeypatch.setattr("cipherlab.rag.retriever.load_bm25", lambda settings: FakeBM25())
    monkeypatch.setattr("cipherlab.rag.retriever.load_dense", lambda settings: dense)
    monkeypatch.setattr(
        "cipherlab.rag.retriever.embed_query_local",
        lambda **kwargs: np.ones(256, dtype=np.float32),
    )

    retriever = RAGRetriever(Settings())
    results = retriever.retrieve("Which S-box should I use?")

    assert [r.chunk_id for r in results] == ["c1"]
    assert results[0].debug["dense_norm"] == 0.0
