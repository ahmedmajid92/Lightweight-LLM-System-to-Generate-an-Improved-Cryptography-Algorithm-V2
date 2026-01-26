# RC6

## High-level summary
- **Architecture:** ARX-like with integer multiplication (AES candidate)
- **Block size:** 128 bits (in AES candidate profile)
- **Key sizes:** 128 / 192 / 256 bits (common)
- **Rounds:** 20 (AES candidate profile)

RC6 builds on RC5 ideas and adds integer multiplication to create additional mixing.

## Structure at a glance
- Operates on four 32-bit words (for the 128-bit variant).
- Each round uses:
  - additions mod 2^32
  - XOR
  - rotations
  - multiplication to generate data-dependent rotation amounts

## Key schedule notes
- Similar in spirit to RC5: expands the key into round subkeys using constants and mixing.

## Why it matters for this project
- Provides an example of ARX + multiplication (not just add/xor/rotate).
- Useful for component registries that want to include “multiply-based mixing” blocks.

## Practical caveats
- As with RC5, parameters matter. Reduced-round variants may have published attacks.
- Implementations must handle 32-bit word overflow behavior precisely.

## References
- RSB+98, *The RC6 Block Cipher* (MIT): https://people.csail.mit.edu/rivest/pubs/RSB+98.pdf
