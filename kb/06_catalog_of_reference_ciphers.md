# Reference cipher catalog (for KB grounding)

This file is a **high-level catalog** of common block ciphers.  
Use it for retrieval grounding, not as a claim of completeness.

## 128-bit block ciphers (common)
- **AES (Rijndael-128)**: SPN, 10/12/14 rounds depending on key size.
- **Twofish**: Feistel-like with complex key-dependent S-boxes, 16 rounds.
- **Serpent**: SPN, 32 rounds, designed with conservative security margin.
- **Camellia**: Feistel, 18/24 rounds.
- **SEED**: Feistel, 16 rounds.
- **IDEA**: 64-bit block (but often grouped with classic ciphers).

## 64-bit block ciphers (legacy/embedded)
- **DES**: Feistel, 16 rounds (legacy; 56-bit key).
- **3DES**: DES applied 3 times (legacy; slow).
- **Blowfish**: Feistel, 16 rounds (64-bit block).
- **CAST-128**: Feistel, 12/16 rounds.
- **RC5 / RC6**: ARX-style (add/rotate/xor), parameterized rounds.

## Why keep this catalog
- enables RAG to answer “compare this design to known families”
- helps the model suggest appropriate structures for block sizes and constraints
