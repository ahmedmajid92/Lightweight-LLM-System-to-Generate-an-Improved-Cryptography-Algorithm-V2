# BlockCipher SFT Dataset (v2)

This dataset is designed for supervised fine-tuning on chat-style models that support
JSONL `messages` examples.

It focuses on:
- Structured knowledge-base style summaries of the 12 lightweight/reference block ciphers used in the project (AES, DES, Blowfish, HIGHT, PRESENT, GIFT, SIMON, SPECK, TEA, XTEA, RC5, LEA).
- Generating `CipherSpec` JSON compatible with Crypto Cipher Lab v2.
- Generating `ImprovementPatch` JSON based on local avalanche-style metrics.
- Writing Python code that uses the lab APIs to export cipher modules and run self-tests.

Files:
- train.jsonl
- valid.jsonl

Format:
Each line is a JSON object with a `messages` array using `system`, `user`, and `assistant` roles.

Notes:
- Outputs are written to be cautious and avoid claiming cryptographic security.
- The goal is to make the fine-tuned model reliable for *research workflows* (design iteration + evaluation),
  not to certify production-ready cryptography.
