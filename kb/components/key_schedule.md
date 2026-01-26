# Key schedule principles

A key schedule expands a master key into round subkeys.

## Goals
- **Diffusion:** each master-key bit should influence many subkey bits.
- **Nonlinearity:** avoid linear relations between subkeys when possible.
- **Avoid weak keys / related-key issues:** prevent structured relationships between keys producing structured differences in subkeys.
- **Round separation:** consecutive round keys should not be simple rotations or copies.

## Common patterns
- AES-style: word rotations + S-box + round constants.
- Hash/KDF-style (research prototypes): use a cryptographic hash/KDF to derive per-round keys.
- ARX-style: mixing loops with constants (RC5/RC6).

## Practical notes for this project
For a research lab, a KDF-based schedule is often a good baseline:
- easy to implement
- avoids obvious structural relations
- reproducible when seeded

But:
- a KDF schedule is not the same as a cipher's standardized schedule.
- if you compare to real algorithms, do so honestly.
