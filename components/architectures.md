# Block cipher architectures: SPN vs Feistel vs ARX

This KB note is meant for component-based cipher construction.

## SPN (Substitution–Permutation Network)
**Typical examples:** AES, Serpent (bit-sliced SPN)

Core idea:
- Alternate **nonlinear substitution** layers (S-boxes) with **linear diffusion** layers (permutations / matrices).
- Usually requires *each layer to be invertible* so decryption is efficient.

Common components:
- S-box layer (byte- or nibble-oriented)
- Permutation (bit or byte permutation)
- Linear diffusion (MDS matrix / MixColumns-style)
- Key mixing (XOR/add) + key schedule

Design levers:
- S-box quality (differential uniformity, linear bias)
- Linear layer strength (branch number)
- Round count (diffusion typically grows quickly with rounds)

## Feistel networks
**Typical examples:** DES, Blowfish, Twofish, Camellia, CAST-128, SEED

Core idea:
- Split the state into two halves (L, R). Each round:
  - L, R = R, L XOR F(R, K_r)

Key property:
- The round function **F does not need to be invertible** to make the whole cipher invertible.
- This is why Feistel designs often use complex, non-invertible F-functions.

Common components:
- F-function (may include expansion, S-boxes, diffusion, ARX ops)
- Key schedule generating per-round subkeys
- Optional whitening keys (pre/post XOR)

Design levers:
- Strength of F-function nonlinearity + diffusion
- Subkey derivation and related-key resistance
- Round count (often 16+)

## ARX-style designs
**Typical examples:** RC5 / RC6 (also many modern lightweight ciphers)

Core idea:
- Use only operations that are fast in software:
  - Addition modulo 2^w
  - XOR
  - Rotate (fixed or data-dependent)

Common components:
- Word-oriented state layout (e.g., 4×32-bit)
- Rotation constants or data-dependent rotations
- Per-round constants and subkeys

Design levers:
- Rotation amounts and structure
- Number of rounds
- Key schedule mixing and constants
