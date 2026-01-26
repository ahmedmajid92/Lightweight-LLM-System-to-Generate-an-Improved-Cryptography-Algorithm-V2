# BlockCipher OpenAI SFT Dataset (v2)

This dataset is designed for supervised fine-tuning on OpenAI models that support SFT
(e.g., gpt-4.1-mini / gpt-4.1-nano).

It focuses on:
- Structured knowledge-base style summaries of well-known block ciphers (AES, DES, 3DES, Blowfish, Twofish, Serpent, Camellia, CAST-128, IDEA, SEED, RC5, RC6).
- Generating `CipherSpec` JSON compatible with Crypto Cipher Lab v2.
- Generating `ImprovementPatch` JSON based on local avalanche-style metrics.
- Writing Python code that uses the lab APIs to export cipher modules and run self-tests.

Files:
- train.jsonl
- valid.jsonl

Format:
Each line is a JSON object with a `messages` array in the OpenAI supervised fine-tuning format.

Notes:
- Outputs are written to be cautious and avoid claiming cryptographic security.
- The goal is to make the fine-tuned model reliable for *research workflows* (design iteration + evaluation),
  not to certify production-ready cryptography.
