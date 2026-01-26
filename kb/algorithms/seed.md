# SEED

## High-level summary
- **Architecture:** Feistel network
- **Block size:** 128 bits
- **Key size:** 128 bits
- **Rounds:** 16

SEED is a 128-bit block cipher standardized and described in IETF documentation.

## Structure at a glance
- 128-bit block split into two 64-bit halves.
- Uses a round function built from:
  - 8-bit S-box substitution
  - diffusion via a “G function” (S-box + linear mixing)
- 16 Feistel rounds with key mixing and swapping.

## Key schedule notes
- Generates round subkeys using constants and rotations.
- Designed so that subkeys vary non-trivially across rounds.

## Why it matters for this project
- Provides a Feistel example at 128-bit block size (more modern than 64-bit designs).
- Useful for comparing diffusion behavior across Feistel designs.

## Practical caveats
- As with any block cipher, correct implementation details are critical (S-box tables, constants, byte order).

## References
- RFC 4269, SEED encryption algorithm: https://www.rfc-editor.org/rfc/rfc4269
