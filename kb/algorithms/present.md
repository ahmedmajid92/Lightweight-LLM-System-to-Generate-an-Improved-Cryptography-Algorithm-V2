# PRESENT (ISO/IEC 29192-2 Lightweight SPN)

PRESENT is an ultra-lightweight block cipher standardized as ISO/IEC 29192-2. It is one of the most widely studied lightweight ciphers.

## Key Properties
- **Architecture:** SPN (Substitution-Permutation Network)
- **Block size:** 64 bits | **Key size:** 80 or 128 bits | **Rounds:** 31
- **S-box:** Single 4-bit S-box applied 16 times per round
- **Permutation:** 64-bit bit permutation (no linear diffusion layer)
- **Gate count:** ~1570 GE (among the lowest for any block cipher)

## Relevance to LWC
PRESENT defined the benchmark for ultra-lightweight SPN ciphers. Its 4-bit S-box design influenced GIFT and other subsequent designs. It is a primary reference for RFID and smart card applications.
