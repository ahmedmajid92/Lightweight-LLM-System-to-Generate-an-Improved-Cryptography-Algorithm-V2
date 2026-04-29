# Crypto Cipher Lab v2 — LLM-Guided Lightweight Block Cipher Design

A **research and education** platform for **LLM-guided generation and iterative improvement of lightweight block ciphers** targeting IoT and resource-constrained environments. This tool enables human-in-the-loop refinement of cipher designs using modular components, deterministic evaluation, and AI-powered suggestions, presented in the thesis-facing narrative as `DeepSeek-R1-Distill-Qwen-14B` for reasoning and `DeepSeek-Coder-V2-Lite-Instruct` for code-facing repair.

> **Important:** Nothing here is a proof of security. Improvements are evaluated only against implemented deterministic heuristics (avalanche, SAC, DDT/LAT, roundtrip correctness). Do not use generated ciphers in production.

---

## Key Features

| Feature                              | Description                                                                         |
| ------------------------------------ | ----------------------------------------------------------------------------------- |
| **Visual Cipher Builder**            | Compose SPN, Feistel, or ARX ciphers from 27+ modular components                   |
| **Deterministic Evaluation**         | Roundtrip verification, SAC analysis, S-box DDT/LAT profiling (no API cost)         |
| **Closed-Loop Iterative Improvement**| Generate → preview → apply → evaluate → accept/reject with full history tracking    |
| **Design-Review Copilot**            | Conversational assistant aware of design state, metrics, and iteration history       |
| **AI Feedback Synthesis**            | Improvement suggestions via the reasoning model, with quality and fast fallbacks     |
| **Adaptive Evolution**               | AST-based mismatch detection + sandboxed LLM component mutation                     |
| **Thesis Artifact Generation**       | LaTeX tables, JSON iteration reports, and JSONL dataset export for publication       |
| **Empirical Benchmarking**           | Automated comparison across the configured reasoning, quality, and fast model slots |
| **RAG-Powered KB**                   | Query a cryptography knowledge base (BM25 + optional embeddings)                    |
| **Code Export**                      | Download standalone Python cipher modules with self-tests                            |
| **Fine-Tuned Model**                 | Optional domain-adapted model for cipher-specific responses                          |

---

## Quick Start (Step by Step)

### Step 1: Clone and Install Dependencies

```bash
git clone <your-repo-url>
cd My-New-Project
```

**Using pip:**

```bash
pip install -r requirements.txt
```

**Using Conda (alternative):**

```bash
conda env create -f environment.yml
conda activate crypto-cipher-lab
```

### Step 2: Configure Local Credentials

Copy the example environment file and fill the credentials and model slots required by your local deployment:

```bash
cp .env.example .env
```

Edit `.env` with the credentials referenced in `.env.example`. Local evaluation (roundtrip, SAC, S-box analysis, export) works without any remote model access.

The thesis-facing model labels used in the docs and explainer UI are:
- Reasoning model: `DeepSeek-R1-Distill-Qwen-14B`
- Code model: `DeepSeek-Coder-V2-Lite-Instruct`

### Step 3: Build the Knowledge Base Index

```bash
python scripts/build_kb_index.py
```

This creates:
- `kb_index/chunks.jsonl` - Chunked knowledge base
- `kb_index/bm25.json` - BM25 search index
- (optional) `kb_index/embeddings.npy` - Dense embeddings if `RAG_USE_EMBEDDINGS=true`

### Step 4: Run the Application

```bash
streamlit run app/streamlit_app.py
```

The app opens in your browser at `http://localhost:8501`.

### Step 5: Use the Application

1. **Choose an architecture** (SPN, Feistel, or ARX) and select components
2. **Run local metrics** to evaluate avalanche properties (free, no API)
3. **Run advanced evaluation** for roundtrip verification, SAC analysis, and S-box profiling
4. **Ask for AI improvements** (requires API key) to get structured suggestions
5. **Apply patches** and compare before/after metrics
6. **Export** the cipher as a standalone Python module
7. **Chat with the KB** to learn about lightweight cryptography

---

## Running the Benchmarks (Phase 5)

The benchmark runner compares LLM model performance at cipher improvement:

```bash
# Quick test: 1 algorithm, 1 repetition, 2 iterations
python scripts/run_benchmarks.py --algorithms AES --reps 1 --max-iterations 2

# Full suite: all 12 algorithms x 3 models x 3 repetitions
python scripts/run_benchmarks.py

# Subset of algorithms
python scripts/run_benchmarks.py --algorithms AES SPECK DES PRESENT

# Regenerate LaTeX tables from existing results
python scripts/run_benchmarks.py --latex-only benchmarks/2026-02-22T.../results.json
```

**Output:** The benchmark creates a timestamped directory under `benchmarks/` containing:

```
benchmarks/2026-02-22T14-30-00Z/
  results.json              # Full experiment data
  tables/
    model_comparison.tex    # Table 1: Model vs model
    architecture_comparison.tex
    algorithm_detail.tex
    convergence_AES.tex     # Per-algorithm convergence curves
    token_cost.tex
  dataset.jsonl             # Compact JSONL (1 line per experiment)
  dataset_full.jsonl        # Detailed JSONL (includes iteration history)
```

**Requirements:** Configure whichever model slots you want to compare before running the benchmark suite. Any unavailable slot is recorded as a failed experiment and the run continues.

---

## Fine-Tuning (Optional)

The repository includes tools to create and fine-tune a custom model on cipher-specific data.

### When to Re-Fine-Tune

Re-fine-tuning is **required** if you:
- Add new algorithms to `AlgorithmsBlock.py` (the SFT dataset references the 12 algorithm library)
- Add new components to `Components.py` (the dataset teaches the model about available components)
- Change the `CipherSpec` or `ImprovementPatch` schemas
- Want the fine-tuned model to know about new capabilities

Re-fine-tuning is **NOT required** if you:
- Only run benchmarks or evaluations
- Use the base reasoning / quality / fast model slots without fine-tuning
- Only modify the evaluation or evolution modules

### Step-by-Step Fine-Tuning Process

**1. Regenerate the SFT dataset** (if algorithms/components changed):

```bash
python scripts/generate_sft_dataset.py
```

This creates 450 training + 50 validation examples under `data/sft/` covering:
- CipherSpec generation for all 12 algorithms
- ImprovementPatch suggestions
- Python code generation
- KB Q&A about cipher concepts

If you added new algorithms, edit the `ALGORITHMS` dict at the top of `scripts/generate_sft_dataset.py` first to include your new algorithms and their specs.

**2. Run the fine-tuning launcher in `scripts/`:**

This step validates `data/sft/train.jsonl` and `data/sft/valid.jsonl`, submits them to your configured backend, monitors the job, and stores the returned fine-tuned model identifier in `.env`.

**3. Verify the fine-tuned model with the validation script in `scripts/`.**

**4. Use it in the app by pointing one of your local model slots at the returned fine-tuned model identifier.**

### Current SFT Dataset

| File          | Examples | Content                                           |
| ------------- | -------- | ------------------------------------------------- |
| `train.jsonl` | 450      | CipherSpec, ImprovementPatch, code, Q&A examples  |
| `valid.jsonl` | 50       | Validation split (different seeds)                 |

---

## Project Structure

```
My-New-Project/
+-- app/
|   +-- streamlit_app.py           # Main Streamlit application (6-stage UI)
+-- cipherlab/                      # Core cipher library package
|   +-- cipher/
|   |   +-- builder.py             # SPN, Feistel, ARX cipher builders
|   |   +-- components_builtin.py  # All cipher components
|   |   +-- cryptanalysis.py       # DDT/LAT, Hamming distance, bit-flip utilities
|   |   +-- exporter.py            # Python code generator
|   |   +-- metrics.py             # Avalanche scoring + evaluate_full()
|   |   +-- registry.py            # Component registry
|   |   +-- spec.py                # CipherSpec and ImprovementPatch (Pydantic)
|   |   +-- validator.py           # Specification validator
|   +-- evaluation/
|   |   +-- roundtrip.py           # P = D(E(P,K),K) verification (Phase 3)
|   |   +-- avalanche.py           # SAC per-bit analysis (Phase 3)
|   |   +-- sbox_analysis.py       # DDT/LAT S-box profiling (Phase 3)
|   |   +-- report.py              # EvaluationReport aggregation (Phase 3)
|   |   +-- feedback.py            # Diagnostic parser + reasoning-model feedback (Phase 3)
|   |   +-- benchmark_runner.py    # Automated model comparison orchestrator (Phase 5)
|   |   +-- latex_exporter.py      # JSON -> LaTeX tables (Phase 5)
|   |   +-- dataset_exporter.py    # JSON -> JSONL dataset (Phase 5)
|   |   +-- iteration_latex.py     # Iteration history LaTeX tables (Phase 6)
|   +-- evolution/
|   |   +-- ast_analyzer.py        # AST dependency mapping + mismatch detection (Phase 4)
|   |   +-- component_mutator.py   # Reasoning-guided component rewriting (Phase 4)
|   |   +-- dynamic_loader.py      # Sandboxed compilation + registry injection (Phase 4)
|   +-- llm/
|   |   +-- assistant.py           # AI improvement suggestions
|   +-- rag/
|   |   +-- retriever.py           # Hybrid BM25 + dense retrieval
|   +-- utils/
|   |   +-- repro.py               # JSON I/O, timestamps, run directories
|   +-- iteration.py               # IterationRecord, IterationHistory, MetricsSummary (Phase 6)
|   +-- config.py                   # Settings (API keys, model names, paths)
|   +-- context_logger.py          # Cipher state capture + copilot context builder
+-- data/sft/                       # Fine-tuning dataset
|   +-- train.jsonl                # 450 training examples
|   +-- valid.jsonl                # 50 validation examples
+-- kb/                             # Built-in knowledge base documents
+-- scripts/
|   +-- build_kb_index.py          # Build RAG search index
|   +-- generate_sft_dataset.py    # Generate fine-tuning data
|   +-- check_finetune_status.py   # Check fine-tuning progress
|   +-- test_finetuned_model.py    # Test fine-tuned model
|   +-- run_benchmarks.py          # Phase 5 benchmark CLI
+-- tests/
|   +-- conftest.py                # pytest configuration (excludes scripts/)
|   +-- test_roundtrip.py          # 15 roundtrip tests (3 hand-crafted + 12 parametrized)
|   +-- test_iteration.py          # Iteration history serialization, rollback, metric deltas
+-- AlgorithmsBlock.py             # 12 LWC algorithm implementations + templates
+-- Components.py                  # 27+ cipher component functions + ComponentRegistry
+-- .env.example                   # Environment variable template
+-- requirements.txt               # Python dependencies
```

---

## Supported Algorithms (12 Lightweight Block Ciphers)

### SPN (Substitution-Permutation Network)

| Algorithm    | Block Size | Key Size | Rounds | Notes                              |
| ------------ | ---------- | -------- | ------ | ---------------------------------- |
| **AES**      | 128-bit    | 128-bit  | 10     | Universal benchmark                |
| **PRESENT**  | 64-bit     | 128-bit  | 31     | ISO/IEC 29192-2, ~1570 GE          |
| **GIFT**     | 128-bit    | 128-bit  | 40     | Improved PRESENT design             |

### Feistel Network

| Algorithm    | Block Size | Key Size | Rounds | Notes                              |
| ------------ | ---------- | -------- | ------ | ---------------------------------- |
| **DES**      | 64-bit     | 128-bit  | 16     | Legacy reference                   |
| **Blowfish** | 64-bit     | 128-bit  | 16     | Key-dependent S-boxes              |
| **HIGHT**    | 64-bit     | 128-bit  | 32     | ISO/IEC 18033-4, RFID/IoT          |
| **TEA**      | 64-bit     | 128-bit  | 64     | Minimal gate count                 |
| **XTEA**     | 64-bit     | 128-bit  | 64     | Improved TEA key schedule          |
| **SIMON**    | 64-bit     | 128-bit  | 42     | NSA LWC, hardware-optimized        |

### ARX (Add-Rotate-XOR)

| Algorithm | Block Size | Key Size | Rounds | Notes                              |
| --------- | ---------- | -------- | ------ | ---------------------------------- |
| **SPECK** | 64-bit     | 128-bit  | 27     | NSA LWC, software-optimized        |
| **RC5**   | 64-bit     | 128-bit  | 12     | Data-dependent rotations           |
| **LEA**   | 128-bit    | 128-bit  | 24     | Korean standard, ARM-optimized     |

---

## Available Components (27+)

### Key Schedules (3)

- `ks.sha256_kdf` - SHA-256 based KDF (universal)
- `ks.des_style` - DES-style rotation and permutation
- `ks.blowfish_style` - Blowfish P-array initialization

### S-boxes (7)

- `sbox.aes` - AES 8-bit S-box (GF(2^8) inverse + affine)
- `sbox.present` - PRESENT 4-bit S-box
- `sbox.gift` - GIFT 4-bit S-box
- `sbox.des` - DES S-boxes (S1-S8)
- `sbox.blowfish` - Blowfish key-dependent S-boxes
- `sbox.serpent` - Serpent 4-bit S-boxes
- `sbox.identity` - No substitution (testing baseline)

### F-functions (4, for Feistel)

- `sbox.tea_f` - TEA F-function
- `sbox.xtea_f` - XTEA F-function
- `sbox.simon_f` - SIMON F-function
- `sbox.hight_f` - HIGHT F-function

### Permutations (6)

- `perm.aes_shiftrows` - AES ShiftRows
- `perm.present` - PRESENT 64-bit bit permutation
- `perm.gift` - GIFT 128-bit bit permutation
- `perm.des_ip` - DES Initial Permutation
- `perm.serpent` - Serpent bit permutation
- `perm.identity` - No permutation

### Linear Layers (3)

- `linear.aes_mixcolumns` - AES MixColumns (GF(2^8))
- `linear.twofish_mds` - Twofish MDS matrix
- `linear.identity` - No mixing

### ARX Operations (4)

- `arx.add_mod32` - Modular addition (32-bit words)
- `arx.rotate_left_3` - RC5/SPECK-style rotation
- `arx.rotate_left_5` - LEA-style rotation
- `arx.mul_mod16` - IDEA multiplication mod 2^16+1

---

## User Interface Guide

The Streamlit application is organized into 6 sections:

### Sidebar: Settings Panel

Configure local credentials, choose the model slots used by the workflow, and set RAG parameters such as top-k chunks and hybrid alpha.

### Section 1: Choose Architecture and Components

Select SPN/Feistel/ARX, configure block size, key size, rounds, and pick components from the registry. Validation status is shown immediately.

### Section 2: Evaluate Locally (No API Cost)

Run avalanche tests to measure plaintext and key sensitivity. Heuristic issues are flagged automatically.

### Section 3: Advanced Evaluation (SAC + S-box Analysis)

- **Roundtrip tests**: Verifies correctness by checking that P = D(E(P,K),K) holds for 1000+ random plaintext/key vectors. If decryption does not perfectly recover every plaintext, the cipher has a functional bug.
- **SAC (Strict Avalanche Criterion) analysis**: Measures diffusion quality. For each input bit, the test flips that single bit and checks whether each output bit changes with probability ~0.5. A well-designed cipher should have SAC deviation close to 0 (ideal mean = 0.5, ideal deviation = 0.0). High deviation indicates weak diffusion — some input bits do not sufficiently influence the output.
- **S-box profiling**:
  - **DDT max (Difference Distribution Table)**: Measures resistance to *differential cryptanalysis*. The DDT records how often a given input difference produces a given output difference through the S-box. A lower DDT max means the S-box spreads input differences more uniformly, making differential attacks harder. An ideal n-bit S-box has DDT max = 2 (for n >= 4). A DDT max equal to the S-box size (e.g., 256 for 8-bit) indicates a completely linear/broken S-box.
  - **LAT max (Linear Approximation Table)**: Measures resistance to *linear cryptanalysis*. The LAT records the correlation between linear combinations of input bits and output bits. A lower LAT max means the S-box resists linear approximations better. For an ideal 8-bit S-box, the LAT max should be as low as possible (AES achieves LAT max = 16 out of 128).
  - **Bijectivity**: Checks whether the S-box is a one-to-one mapping (every input maps to a unique output). A bijective S-box is invertible, which is required for decryption. Non-bijective S-boxes lose information and cannot be reversed.
- **I/O compatibility check**: AST-based mismatch detection verifies that each component's input/output sizes are compatible with adjacent pipeline stages (e.g., the permutation output width matches the linear layer input width).

### Section 4: Iterative Improvement (Closed-Loop Workflow)

LLM-guided generate → preview → apply → evaluate → accept/reject loop:

1. **Generate patch** — The reasoning model analyzes current metrics and diagnostics to propose an `ImprovementPatch` (component swaps, round changes, rationale).
2. **Preview** — Inspect the proposed changes, model reasoning trace, and diff before committing.
3. **Apply & evaluate** — The patch is applied to a staged working spec; before/after metrics are computed and displayed side-by-side.
4. **Accept / reject** — Accept to advance the working design, or reject to discard. Every decision is recorded with full traceability (spec snapshots, metrics, model, seed, timestamp, decision reason).
5. **History & rollback** — View the full iteration history table. Roll back to any previously accepted iteration if needed.

### Section 5: Export & Thesis Artifacts

- **Python module** — Download a standalone encrypt/decrypt module with self-test.
- **Reproducible run** — Save full run state (spec, metrics, iteration history, RAG context, LLM transcript) to `runs/`.
- **JSON export** — Download the complete iteration history as structured JSON for later analysis.
- **LaTeX tables** — Auto-generated publication-ready tables (iteration summary, accepted metric deltas, aggregate statistics) using booktabs formatting. Preview and download individual `.tex` files.

### Section 6: Design-Review Copilot

Conversational assistant aware of the current working design, evaluation results, diagnostics, iteration history, and pending patches. Uses tiered context injection so the LLM always knows what you're working on. The RAG knowledge base provides supporting evidence from lightweight cryptography literature, but the copilot's primary role is design review — discussing tradeoffs, suggesting next steps, and helping justify decisions for thesis writing.

---

## Architecture Overview (6 Phases)

| Phase | Name | Key Capability |
|-------|------|----------------|
| 1 | Model Gateway | Configured reasoning, quality, and fast model slots behind a common interface |
| 2 | Component Optimization | 27+ mathematically accurate components, 12 LWC algorithms |
| 3 | Deterministic Evaluation | Roundtrip, SAC, DDT/LAT — all local, no API cost |
| 4 | Adaptive Evolution | AST mismatch detection, LLM component mutation, sandboxed loading |
| 5 | Empirical Benchmarking | Automated model comparison, LaTeX tables, JSONL dataset export |
| 6 | Closed-Loop Refinement | Iterative improvement with full traceability, design-review copilot, thesis artifact generation |

---

## Thesis Contributions

This platform supports a thesis on **LLM-guided generation and iterative improvement of lightweight block ciphers** with human-in-the-loop refinement. Key research contributions enabled by the system:

1. **Reproducible improvement trajectories** — Every iteration is recorded with before/after spec snapshots, metric deltas, model identity, reasoning trace, seed, and accept/reject decision. This provides an auditable chain of evidence for how a cipher design evolved.
2. **Quantitative before/after justification** — Side-by-side metric comparison (avalanche, SAC, overall score) gives concrete numeric evidence for each design change.
3. **Structured export for analysis** — Full iteration history is exportable as JSON (for programmatic analysis) and LaTeX tables (for direct inclusion in thesis chapters).
4. **Human-in-the-loop refinement** — The accept/reject gate ensures all design decisions remain under researcher control, with the LLM acting as an advisor rather than an autonomous agent.
5. **Design-review copilot** — Conversational assistant grounded in the current design state and evaluation results, supporting iterative reasoning about tradeoffs and next steps.

---

## Security Disclaimer

1. **Local avalanche tests are NOT cryptanalysis** - They cannot detect subtle weaknesses
2. **Do not interpret any score as security** - A high score does not mean the cipher is secure
3. **Generated ciphers are for research only** - Never use in production systems
4. **Consult cryptographers** - Before publishing any cipher-related research

---

## License

This project is for educational and research purposes only.
