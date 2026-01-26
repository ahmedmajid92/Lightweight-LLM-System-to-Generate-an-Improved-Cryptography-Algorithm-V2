# Twofish

## High-level summary
- **Architecture:** Feistel network (AES finalist)
- **Block size:** 128 bits
- **Key sizes:** 128 / 192 / 256 bits
- **Rounds:** 16
- **Notable design features:** whitening + key-dependent S-boxes + strong linear diffusion

Twofish was an AES finalist designed for strong security margins and practical performance.

## Structure at a glance
Twofish uses:
- Input and output **whitening** (XOR with subkeys before/after the Feistel core)
- A 16-round Feistel network where the F-function uses:
  - Key-dependent S-boxes
  - An MDS matrix for diffusion
  - A pseudo-Hadamard transform (PHT)
  - A key schedule generating subkeys for whitening and round operations

## Why it matters for this project
- Provides a reference for mixing:
  - classical Feistel structure
  - strong diffusion via an explicit MDS layer
  - whitening as an extra “degree of freedom”
- A good template when your generator wants “AES-level” block size (128) but prefers a Feistel core.

## Practical caveats
- Implementation details (byte order, MDS tables, key-dependent S-box derivation) matter a lot.
- For a *toy* generator, it’s easy to copy the “shape” of Twofish but miss crucial security details.

## References
- Twofish project page (Schneier et al.): https://www.schneier.com/academic/twofish/
- NIST AES project (historical context): https://csrc.nist.gov/projects/aes
