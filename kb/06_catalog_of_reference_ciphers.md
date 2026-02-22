# Lightweight Cryptography (LWC) Reference Cipher Catalog

This file is a **high-level catalog** of the 12 baseline lightweight block ciphers.
Use it for retrieval grounding in IoT and resource-constrained cryptography research.

## SPN Ciphers
- **AES (Rijndael-128)**: SPN, 128-bit blocks, 10 rounds. Universal benchmark.
- **PRESENT**: SPN, 64-bit blocks, 31 rounds. ISO/IEC 29192-2 LWC standard (~1570 GE).
- **GIFT-128**: SPN, 128-bit blocks, 40 rounds. Improved PRESENT design, NIST LWC finalist basis.

## Feistel Ciphers
- **DES**: Feistel, 64-bit blocks, 16 rounds. Legacy reference (56-bit key originally).
- **Blowfish**: Feistel, 64-bit blocks, 16 rounds. Key-dependent S-boxes.
- **HIGHT**: Generalized Feistel, 64-bit blocks, 32 rounds. ISO/IEC 18033-4 for RFID/IoT.
- **TEA**: Feistel, 64-bit blocks, 64 rounds. Minimal gate count, no S-boxes.
- **XTEA**: Feistel, 64-bit blocks, 64 rounds. Improved TEA key schedule.
- **SIMON 64/128**: Feistel (AND-rotate-XOR), 64-bit blocks, 42 rounds. NSA LWC, hardware-optimized.

## ARX Ciphers
- **SPECK 64/128**: ARX, 64-bit blocks, 27 rounds. NSA LWC, software-optimized.
- **RC5**: ARX, 64-bit blocks, 12 rounds. Data-dependent rotations.
- **LEA**: ARX, 128-bit blocks, 24 rounds. Korean standard, ARM-optimized.

## Why this catalog
- Enables RAG to answer "compare this design to known LWC families"
- Helps the model suggest appropriate lightweight structures for IoT constraints
- Covers hardware-optimized (SIMON, PRESENT, GIFT), software-optimized (SPECK, LEA), and minimal-complexity (TEA, XTEA) design paradigms
