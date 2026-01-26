# Camellia

## High-level summary
- **Architecture:** Feistel network with additional FL / FL⁻¹ layers
- **Block size:** 128 bits
- **Key sizes:** 128 / 192 / 256 bits
- **Rounds:** 18 (128-bit key) or 24 (192/256-bit key)

Camellia is a modern block cipher standardized in multiple environments and described in IETF RFCs.

## Structure at a glance
- 128-bit block split into two 64-bit halves.
- Uses a Feistel core with an F-function that combines S-box substitution and linear diffusion.
- Inserts **FL** and **FL⁻¹** functions at fixed points to strengthen security and complicate certain attack classes.

## Key schedule notes
- Derives subkeys via rotations and constants.
- Produces keys for the Feistel rounds plus the FL/FL⁻¹ layers and whitening.

## Why it matters for this project
- Shows how to extend a Feistel core with additional keyed functions (FL layers).
- Useful for exploring “round-structure diversity” rather than repeating the same round function only.

## Practical caveats
- Like other mature designs, details are critical (exact S-boxes, constants, placement of FL layers).
- A component-based generator can imitate the architecture, but security requires careful parameterization and review.

## References
- RFC 3713, Camellia encryption algorithm: https://www.rfc-editor.org/rfc/rfc3713
