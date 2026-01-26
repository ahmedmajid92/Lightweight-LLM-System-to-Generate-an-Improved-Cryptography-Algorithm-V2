# DES (Data Encryption Standard)

## High-level summary
- **Architecture:** Feistel network
- **Block size:** 64 bits
- **Key size:** 56 bits effective (64 bits with parity)
- **Rounds:** 16

DES is a historic standard. It is useful as a canonical **Feistel** example but not suitable for modern security due to short key length.

## Structure at a glance
DES splits the 64-bit block into two 32-bit halves (L, R). Each round:
- Expands R from 32 → 48 bits (E expansion)
- XORs with a 48-bit round key
- Applies 8 S-boxes (each 6 → 4 bits)
- Applies a permutation (P)
- XORs with L, then swaps halves (Feistel structure)

## Key schedule notes
- Round keys are derived using:
  - PC-1/PC-2 permutations
  - Left-rotations of two 28-bit key halves
- Generates 16 subkeys of 48 bits.

## Why it matters for this project
DES is a strong reference for:
- Feistel invertibility (round function does **not** need to be invertible).
- A classic “expand → substitute → permute” round function design.
- Studying how small components (S-boxes + permutations) drive diffusion.

## Practical caveats
- **56-bit keys are brute-forceable** with modern resources.
- **64-bit blocks** hit birthday-bound limits sooner than 128-bit blocks; large-volume encryption can be risky even with a strong key.

## References
- NIST FIPS 46-3 (DES archive): https://csrc.nist.gov/publications/detail/fips/46/3/archive/1999-10-25
