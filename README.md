# Crypto Cipher Lab v2 (OpenAI edition)

This repo is a **research / education** sandbox for experimenting with *block-cipher-like* constructions.

It lets you:
- Build a cipher spec (SPN or Feistel) from modular components
- Run local metrics (avalanche tests) without any API cost
- Retrieve relevant crypto notes from a local KB (BM25 + optional embeddings)
- Ask an OpenAI model to suggest small improvements (structured output patch)
- Export a standalone Python module that implements the cipher

**Important:** Nothing here is a proof of security. Do not use generated ciphers in production.

---

## 1) Setup

### 1.1 Install dependencies
```bash
pip install -r requirements.txt
```

### 1.2 Set environment variables
Copy `.env.example` to `.env` and fill your values.

Required:
- `OPENAI_API_KEY`

Optional:
- `OPENAI_MODEL_FAST`
- `OPENAI_MODEL_QUALITY`
- `OPENAI_EMBEDDING_MODEL`
- `RAG_USE_EMBEDDINGS=true` (if you want hybrid retrieval)

---

## 2) Build the KB index (RAG)

The project reads Markdown/PDF/TXT files from:
- `kb/` (built-in KB)
- `kb_external/` (your extra PDFs / docs)

Build the index:
```bash
python scripts/build_kb_index.py
```

This creates:
- `kb_index/chunks.jsonl`
- `kb_index/bm25.json`
- (optional) `kb_index/embeddings.npy` and `dense_ids.json` when `RAG_USE_EMBEDDINGS=true`

---

## 3) Run the app

```bash
streamlit run app/streamlit_app.py
```

---

## 4) Export cipher code

In the Streamlit UI:
- configure a cipher
- click “Generate Python module”
- download the file (it includes a `self_test()`)

---

## 5) Fine-tuning (optional)

This repo includes an OpenAI SFT dataset under `data/sft/`:
- `train.jsonl`
- `valid.jsonl`

It contains examples for:
- cipher KB summaries
- generating CipherSpec JSON
- generating ImprovementPatch JSON
- exporter usage code

You can fine-tune a supported OpenAI model (check the OpenAI docs for current availability).
In many cases, RAG + Structured Outputs is enough and cheaper than fine-tuning.

---

## 6) Security disclaimer

- Local avalanche tests are **not** cryptanalysis.
- Do not interpret any “score” as security.
- If you plan to publish a thesis, include clear warnings and cite standard references.

