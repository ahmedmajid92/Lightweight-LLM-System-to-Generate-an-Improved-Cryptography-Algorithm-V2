# Hybrid retrieval for block-cipher RAG

This project uses **hybrid retrieval** to keep LLM calls *cheap* and *reliable*:

- **Lexical retrieval (BM25)** is strong when the query contains exact terms:
  - algorithm names (AES, Twofish, Camellia)
  - component names (S-box, MDS, key schedule)
  - attack terms (differential, linear, integral)
- **Dense retrieval (local embeddings + FAISS)** is strong when the query is phrased loosely:
  - “how to improve diffusion”
  - “why is avalanche low”
  - “common mistakes in key schedules”

## Why hybrid > dense-only for cryptography
Block-cipher literature is heavy on *terminology*. Dense models sometimes miss exact technical keywords.
BM25 anchors retrieval to those keywords, while dense retrieval helps with paraphrases.

## Score fusion
We retrieve top candidates from BM25 and dense search, then normalize scores and fuse:

```
score = w_dense * norm_dense + w_lex * norm_bm25
```

Weights are controlled by:
- `RAG_DENSE_WEIGHT`
- `RAG_LEX_WEIGHT`

## Practical advice
- If your KB is mostly structured docs with many exact terms → increase BM25 weight.
- If your KB has longer natural-language explanations → increase dense weight.
- Always keep **top_k** small (e.g., 6–10) to reduce context tokens.
