# GIFT (Lightweight SPN Cipher)

GIFT is a lightweight block cipher that improves upon PRESENT's design, offering better security per gate equivalent.

## Key Properties
- **Architecture:** SPN (Substitution-Permutation Network)
- **Variants:** GIFT-64 (64-bit blocks) and GIFT-128 (128-bit blocks)
- **GIFT-128:** 128-bit blocks, 128-bit key, 40 rounds
- **S-box:** 4-bit S-box (different from PRESENT's)
- **Permutation:** Optimized bit permutation for better diffusion
- **Gate count:** ~1700 GE (GIFT-64)

## Relevance to LWC
GIFT represents the evolution of PRESENT-style SPN ciphers. Its improved diffusion layer achieves better security margins with similar hardware cost. GIFT-COFB (based on GIFT-128) was a finalist in the NIST Lightweight Cryptography competition.
