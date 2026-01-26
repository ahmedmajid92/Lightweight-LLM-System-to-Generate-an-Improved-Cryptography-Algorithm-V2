# IDEA (International Data Encryption Algorithm)

## High-level summary
- **Architecture:** “Mixed algebraic groups” design (Lai–Massey-like structure with MA layers)
- **Block size:** 64 bits
- **Key size:** 128 bits
- **Rounds:** 8.5 (8 full rounds + output transform)

IDEA mixes operations from different algebraic groups to resist attacks that exploit a single algebraic structure.

## Structure at a glance
IDEA operates on four 16-bit words and combines:
- **XOR** (bitwise)
- **Addition mod 2^16**
- **Multiplication mod (2^16 + 1)** (with a special mapping for zero)

These mixed operations create nonlinearity and diffusion without using large S-box tables.

## Key schedule notes
- Expands the 128-bit key into 52 16-bit subkeys.
- Uses cyclic rotations of the key material to generate subkeys.

## Why it matters for this project
- IDEA is a strong example that “block cipher components” are not only S-boxes and permutations.
- Useful for generating alternatives that use modular arithmetic and algebraic mixing.

## Practical caveats
- 64-bit block size implies birthday-bound limits at high data volumes.
- Implementations must handle modular multiplication carefully (including the special case mapping for 0).

## References
- Lai & Massey, *On the Design and Security of Block Ciphers* (ETH Zürich): https://www.research-collection.ethz.ch/bitstream/handle/20.500.11850/68861/eth-23109-01.pdf
- RFC 5469 (IDEA in IPsec): https://www.rfc-editor.org/rfc/rfc5469
