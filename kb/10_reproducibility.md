# Reproducibility checklist

A PhD-quality system should allow others to reproduce results.

## What to record for each run
- Cipher specification (`spec.json`)
- Generated code (`cipher.py`)
- Test battery outputs (`report.json`)
- Random seed
- Model label + version used for the run record (for example `DeepSeek-R1-Distill-Qwen-14B` or `DeepSeek-Coder-V2-Lite-Instruct`)
- Retrieval configuration (top_k, weights, embedding model name)
- KB snapshot metadata (which KB files were indexed)

## This project
Every build/test run writes into `./runs/<timestamp>_<label>_<tag>/`:
- `env.json`
- `spec.json`
- `cipher.py` (when built)
- `report.json` (when tests run)

## Suggested additions (optional)
- Git commit hash of your codebase
- Hash of KB folder contents (to detect KB drift)
- Prompt/response logs for LLM calls (with redactions if needed)
