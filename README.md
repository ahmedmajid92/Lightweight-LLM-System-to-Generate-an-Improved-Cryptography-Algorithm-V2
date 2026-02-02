# 🔐 Crypto Cipher Lab v2 (OpenAI Edition)

A **research and education** sandbox for experimenting with block cipher constructions. This tool allows you to compose, analyze, and iterate on cipher designs using modular components and AI-powered suggestions.

> ⚠️ **Important:** Nothing here is a proof of security. Do not use generated ciphers in production.

---

## ✨ Key Features

| Feature                   | Description                                                      |
| ------------------------- | ---------------------------------------------------------------- |
| **Visual Cipher Builder** | Compose SPN, Feistel, or ARX ciphers from modular components     |
| **Local Metrics**         | Run avalanche tests without any API cost                         |
| **RAG-Powered KB**        | Query a cryptography knowledge base (BM25 + optional embeddings) |
| **AI Improvements**       | Get structured improvement suggestions from OpenAI               |
| **Code Export**           | Download standalone Python modules with self-tests               |
| **Fine-Tuned Model**      | Uses a custom fine-tuned model for cipher-specific responses     |

---

## 🖥️ User Interface Guide

The Streamlit application is organized into **5 main sections** plus a **sidebar** for configuration.

### 📱 Sidebar: Settings Panel

The sidebar contains two configuration sections:

#### OpenAI Settings

| Setting                   | Description                                                  |
| ------------------------- | ------------------------------------------------------------ |
| **OPENAI_API_KEY**        | Your OpenAI API key (required for AI features)               |
| **Fast model**            | Model used for quick responses (default: `gpt-4.1-mini`)     |
| **Quality model**         | Model used for complex tasks (default: `gpt-4.1`)            |
| **Model for improvement** | Select which model to use for generating improvement patches |

#### RAG Settings

| Setting             | Description                                                                   |
| ------------------- | ----------------------------------------------------------------------------- |
| **Top-k KB chunks** | Number of knowledge base chunks to retrieve (2-12)                            |
| **Hybrid alpha**    | Balance between dense embeddings and BM25 (0.0 = BM25 only, 1.0 = dense only) |

---

### 📐 Section 1: Choose Architecture and Components

This is where you design your cipher by selecting its fundamental building blocks.

#### Architecture Selection

Choose from three cipher architectures:

- **SPN (Substitution-Permutation Network)**: Used by AES, Serpent. Applies substitution, permutation, and linear mixing in each round.
- **FEISTEL**: Used by DES, Blowfish, Twofish. Splits the block in half and applies a round function.
- **ARX (Add-Rotate-XOR)**: Used by RC5, RC6, IDEA. Uses only modular addition, rotation, and XOR.

#### Left Column - Basic Parameters

| Parameter       | Description                                                           |
| --------------- | --------------------------------------------------------------------- |
| **Cipher name** | A custom name for your cipher design (e.g., "MyCipherV2")             |
| **Seed**        | Integer for reproducibility - same seed produces identical round keys |
| **Rounds**      | Number of encryption rounds (more rounds = more security, slower)     |

#### Right Column - Block and Key Size

| Parameter             | Description                                     |
| --------------------- | ----------------------------------------------- |
| **Block size (bits)** | Size of data processed at once (64 or 128 bits) |
| **Key size (bits)**   | Length of the encryption key (128 or 256 bits)  |

#### Component Dropdowns

Depending on the architecture, you select different components:

**For SPN:**
| Component | Purpose | Example Options |
|-----------|---------|-----------------|
| **S-box** | Non-linear substitution | `sbox.aes`, `sbox.serpent` |
| **Permutation** | Bit/byte reordering | `perm.aes_shiftrows`, `perm.serpent` |
| **Linear diffusion** | Spread bit changes | `linear.aes_mixcolumns`, `linear.twofish_mds` |
| **Key schedule** | Generate round keys | `ks.sha256_kdf`, `ks.des_style` |

**For Feistel:**
| Component | Purpose | Example Options |
|-----------|---------|-----------------|
| **F-function S-box** | Non-linear function | `sbox.aes`, `sbox.des`, `sbox.blowfish` |
| **F-function permutation** | Round function mixing | `perm.identity` |
| **Key schedule** | Generate round keys | `ks.sha256_kdf`, `ks.blowfish_style` |

**For ARX:**
| Component | Purpose | Example Options |
|-----------|---------|-----------------|
| **ARX addition** | Modular addition | `arx.add_mod32`, `arx.mul_mod16` |
| **ARX rotation** | Bit rotation | `arx.rotate_left_3`, `arx.rotate_left_5` |
| **Key schedule** | Generate round keys | `ks.sha256_kdf` |

After configuration, validation status is displayed:

- ✅ **Green**: "Spec looks valid" - ready to proceed
- ❌ **Red**: Validation errors listed - fix before continuing

---

### 📊 Section 2: Evaluate Locally (No API Cost)

Click **"Run local metrics"** to analyze your cipher design without any API calls.

#### Metrics Displayed

| Metric            | Ideal Value | Meaning                                                                  |
| ----------------- | ----------- | ------------------------------------------------------------------------ |
| **pt_avalanche**  | ~0.50       | Plaintext avalanche: % of output bits that change when 1 input bit flips |
| **key_avalanche** | ~0.50       | Key avalanche: % of output bits that change when 1 key bit flips         |
| **overall_score** | 0-100       | Combined quality score based on all metrics                              |

#### Detected Issues

The right column shows heuristic-detected problems:

- ⚠️ **Yellow warnings**: Issues like poor avalanche, weak diffusion
- ✅ **Green success**: "No obvious issues flagged by heuristics"

> **Note**: These are heuristic tests, NOT cryptanalysis. They cannot prove security.

---

### 💾 Section 3: Export Cipher as Python Code

Click **"Generate Python module"** to create a standalone implementation.

#### What You Get

A complete Python file containing:

- The cipher class with `encrypt_block()` and `decrypt_block()` methods
- All required components embedded (S-boxes, permutations, etc.)
- A `self_test()` function for verification
- Full documentation and usage examples

#### Available Actions

| Button                        | Action                                                 |
| ----------------------------- | ------------------------------------------------------ |
| **Download cipher_module.py** | Save the Python file to your computer                  |
| **Save as reproducible run**  | Store the spec, code, and metrics in `runs/` directory |

---

### 🤖 Section 4: Ask for Improvement Suggestions (Uses OpenAI)

This is the AI-powered feature that suggests how to improve your cipher.

#### Prerequisites

- ✅ Valid cipher specification
- ✅ OpenAI API key configured
- ✅ Local metrics have been run

#### Process

1. Click **"Suggest improvements"**
2. System retrieves relevant knowledge from the KB
3. OpenAI generates a structured `ImprovementPatch`

#### The ImprovementPatch Contains

| Field                  | Description                                       |
| ---------------------- | ------------------------------------------------- |
| **summary**            | Brief description of suggested changes            |
| **rationale**          | List of reasons/design principles for the changes |
| **new_rounds**         | Suggested number of rounds (if change needed)     |
| **replace_components** | Component substitutions (e.g., swap S-box)        |
| **add_notes**          | Additional design notes                           |

#### Applying the Patch

Click **"Apply patch and re-evaluate"** to:

1. Create a new spec with suggested changes
2. Run metrics on the improved design
3. Compare before/after performance
4. Download the improved cipher module

---

### 💬 Section 5: KB Chat (Block Ciphers Only)

A conversational interface to query the cryptography knowledge base.

#### How to Use

1. Type a question in the text input (e.g., "What is the avalanche effect?")
2. Click **"Ask"**
3. The system retrieves relevant KB chunks and generates an answer

#### Example Questions

- "What makes a good S-box?"
- "Explain the difference between SPN and Feistel"
- "What is the birthday bound for 64-bit blocks?"
- "How does key whitening improve security?"

#### Retrieved KB Chunks

Expand the **"Retrieved KB chunks"** section to see:

- Source documents used to generate the answer
- Relevance scores for each chunk
- Original text excerpts

---

## 🚀 Quick Start

### 1. Install Dependencies

**Using Conda (Recommended):**

```bash
conda env create -f environment.yml
conda activate crypto-cipher-lab
```

**Using pip:**

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

Copy `.env.example` to `.env` and set:

```env
# Required
OPENAI_API_KEY=sk-your-api-key-here

# Optional (have sensible defaults)
OPENAI_MODEL_FAST=gpt-4.1-mini
OPENAI_MODEL_QUALITY=gpt-4.1
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
RAG_USE_EMBEDDINGS=false
RAG_TOP_K=6
RAG_HYBRID_ALPHA=0.55

# Fine-tuned model (if available)
FINETUNED_MODEL=ft:gpt-4.1-mini-2025-04-14:your-org:cipher-lab:xxxxx
```

### 3. Build the KB Index

```bash
python scripts/build_kb_index.py
```

This creates:

- `kb_index/chunks.jsonl` - Chunked knowledge base
- `kb_index/bm25.json` - BM25 search index
- (optional) `kb_index/embeddings.npy` - Dense embeddings if `RAG_USE_EMBEDDINGS=true`

### 4. Run the Application

```bash
streamlit run app/streamlit_app.py
```

The app opens in your browser at `http://localhost:8501`

---

## 📁 Project Structure

```
My-New-Project/
├── app/
│   └── streamlit_app.py      # Main Streamlit application
├── cipherlab/                 # Core cipher library package
│   ├── cipher/
│   │   ├── builder.py        # SPN, Feistel, ARX cipher builders
│   │   ├── components_builtin.py  # All 22 cipher components
│   │   ├── exporter.py       # Python code generator
│   │   ├── metrics.py        # Avalanche and scoring
│   │   ├── registry.py       # Component registry
│   │   ├── spec.py           # CipherSpec and ImprovementPatch models
│   │   └── validator.py      # Specification validator
│   ├── llm/
│   │   ├── assistant.py      # AI improvement suggestions
│   │   └── openai_provider.py # OpenAI API wrapper
│   └── rag/
│       └── retriever.py      # RAG retrieval system
├── data/sft/                  # Fine-tuning dataset
│   ├── train.jsonl           # 450 training examples
│   └── valid.jsonl           # 48 validation examples
├── kb/                        # Built-in knowledge base
├── scripts/
│   ├── build_kb_index.py     # Build RAG index
│   ├── finetune_openai.py    # Run fine-tuning
│   ├── check_finetune_status.py
│   └── test_finetuned_model.py
├── AlgorithmsBlock.py        # Standalone cipher implementations
├── Components.py             # All cipher component functions
├── .env                      # Environment configuration
└── requirements.txt          # Python dependencies
```

---

## 🔧 Supported Algorithms

The system supports **12 reference block cipher algorithms** across 3 architectures:

### SPN (Substitution-Permutation Network)

| Algorithm   | Block Size | Key Size | Rounds |
| ----------- | ---------- | -------- | ------ |
| **AES**     | 128-bit    | 128-bit  | 10     |
| **Serpent** | 128-bit    | 128-bit  | 32     |

### Feistel Network

| Algorithm    | Block Size | Key Size | Rounds |
| ------------ | ---------- | -------- | ------ |
| **DES**      | 64-bit     | 128-bit  | 16     |
| **3DES**     | 64-bit     | 128-bit  | 48     |
| **Blowfish** | 64-bit     | 128-bit  | 16     |
| **Twofish**  | 128-bit    | 256-bit  | 16     |
| **Camellia** | 128-bit    | 128-bit  | 18     |
| **CAST-128** | 64-bit     | 128-bit  | 16     |
| **SEED**     | 128-bit    | 128-bit  | 16     |

### ARX (Add-Rotate-XOR)

| Algorithm | Block Size | Key Size | Rounds |
| --------- | ---------- | -------- | ------ |
| **RC5**   | 64-bit     | 128-bit  | 12     |
| **RC6**   | 128-bit    | 128-bit  | 20     |
| **IDEA**  | 64-bit     | 128-bit  | 8      |

---

## 📦 Available Components (22 Total)

### Key Schedules (3)

- `ks.sha256_kdf` - SHA-256 based KDF (universal)
- `ks.des_style` - DES-style rotation and permutation
- `ks.blowfish_style` - Blowfish P-array initialization

### S-boxes (5)

- `sbox.aes` - AES 8-bit S-box
- `sbox.des` - DES S-boxes (S1-S8)
- `sbox.blowfish` - Blowfish key-dependent S-boxes
- `sbox.serpent` - Serpent 4-bit S-boxes
- `sbox.identity` - No substitution (testing)

### Permutations (4)

- `perm.aes_shiftrows` - AES ShiftRows
- `perm.des_ip` - DES Initial Permutation
- `perm.serpent` - Serpent bit permutation
- `perm.identity` - No permutation

### Linear Layers (3)

- `linear.aes_mixcolumns` - AES MixColumns
- `linear.twofish_mds` - Twofish MDS matrix
- `linear.identity` - No mixing

### ARX Operations (4)

- `arx.add_mod32` - Modular addition (32-bit words)
- `arx.rotate_left_3` - RC5-style rotation
- `arx.rotate_left_5` - RC6/IDEA-style rotation
- `arx.mul_mod16` - IDEA multiplication mod 2^16+1

---

## 🎓 Fine-Tuning (Optional)

The repository includes a supervised fine-tuning dataset under `data/sft/`:

| File          | Examples | Content             |
| ------------- | -------- | ------------------- |
| `train.jsonl` | 450      | Training examples   |
| `valid.jsonl` | 48       | Validation examples |

### Dataset Coverage

- CipherSpec generation from natural language
- ImprovementPatch suggestions
- Python code generation with comments
- KB question answering
- Algorithm comparisons

### Run Fine-Tuning

```bash
python scripts/finetune_openai.py
```

The script will:

1. Validate the dataset format
2. Upload files to OpenAI
3. Create and monitor the fine-tuning job
4. Update `.env` with the new model ID

---

## ⚠️ Security Disclaimer

1. **Local avalanche tests are NOT cryptanalysis** - They cannot detect subtle weaknesses
2. **Do not interpret any score as security** - A high score does not mean the cipher is secure
3. **Generated ciphers are for research only** - Never use in production systems
4. **Consult cryptographers** - Before publishing any cipher-related research

---

## 📄 License

This project is for educational and research purposes only.

---

## 🤝 Contributing

Contributions are welcome! Please ensure any additions:

- Include proper documentation
- Follow the existing code style
- Add appropriate tests
- Update the README if needed
