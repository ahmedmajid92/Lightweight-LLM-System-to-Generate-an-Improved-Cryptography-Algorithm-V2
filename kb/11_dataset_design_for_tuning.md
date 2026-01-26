# Fine-tuning dataset design (block ciphers)

If you fine-tune a model, the dataset should match how you *use* the model.

## Recommended target behaviors
1) **Patch-the-spec** (best for reliability)
- Input: current spec + metrics + goal
- Output: updated spec JSON (minimal, structured)

2) **Explain improvements with measurable justification**
- Input: proposed change + test results
- Output: explanation referencing avalanche/diffusion/S-box metrics

3) (Optional) **Code generation for toy ciphers**
- Input: spec JSON
- Output: Python code (encrypt/decrypt) using a standard template

## Avoid in training data
- Claims like “this cipher is secure”
- Unverifiable security guarantees
- “Novel” components without invertibility checks

## Data format (Vertex supervised tuning)
Use JSONL records with:
- `systemInstruction`
- `contents` messages with roles `user` and `model`

See Google docs for the canonical JSONL structure.

## Scaling up
To grow beyond synthetic examples:
- Add curated examples from your own lab notebooks:
  - “What went wrong and how I fixed it”
  - “Why this diffusion layer was weak”
- Add examples that reference real ciphers as baselines, but output is *your own* analysis/code.
