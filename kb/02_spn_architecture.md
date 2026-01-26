# SPN architecture template and constraints

An SPN (Substitution–Permutation Network) is usually built from repeated rounds:

1. AddRoundKey (XOR with a round key)
2. Substitution layer (parallel S-boxes)
3. Permutation layer (bit/byte permutation)
4. Linear diffusion layer (matrix multiply / MDS-like)

Often the final round omits diffusion.

## Structural constraints for an SPN builder
- The **overall round function must be invertible**.
- Each chosen component must be either:
  - invertible itself, *or*
  - used in a way that still yields an invertible whole (advanced designs).
- For a modular builder, the simplest rule is:
  - substitution layer must be invertible
  - permutation layer must be invertible
  - diffusion layer must be invertible

## Practical compatibility constraints
- Block size must match component expectations:
  - AES-like components assume a 4×4 byte state (128-bit).
- “MixColumns” is not meaningful unless state layout is defined.
- If you allow mixing components from different ciphers, add metadata:
  - state layout (row-major/column-major)
  - unit size (bit/nybble/byte/word)
  - diffusion matrix dimensions
  - whether components operate on full state or per-lane

## What to optimize
- Avalanche heuristics often improve with:
  - stronger diffusion (higher branch number)
  - more rounds
  - better S-box properties (lower max DDT probability, lower max LAT correlation)
