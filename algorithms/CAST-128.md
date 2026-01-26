# CAST-128 (CAST5)

## High-level summary
- **Architecture:** Feistel network
- **Block size:** 64 bits
- **Key size:** 40 to 128 bits
- **Rounds:** 12 or 16 (depends on key size)

CAST-128 is also known as CAST5 and appears historically in PGP and related tooling.

## Structure at a glance
- 64-bit block split into two 32-bit halves.
- Feistel rounds use different “types” of round functions (mixing operations vary by round), combining:
  - addition/subtraction modulo 2^32
  - XOR
  - rotations
  - S-box lookups

## Key schedule notes
- Produces per-round masking subkeys and rotation subkeys.
- Number of rounds depends on key length (shorter keys → fewer rounds).

## Why it matters for this project
- Good example that “Feistel + S-boxes” is not one fixed design: round functions can vary.
- Provides component examples beyond plain XOR: modular arithmetic + rotations (ARX-like flavor).

## Practical caveats
- 64-bit block size has modern limits for high-volume encryption.
- Primarily legacy today; useful as a reference, not as a modern default.

## References
- RFC 2144, CAST-128: https://www.rfc-editor.org/rfc/rfc2144
