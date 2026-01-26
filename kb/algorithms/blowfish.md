# Blowfish

## High-level summary
- **Architecture:** Feistel network
- **Block size:** 64 bits
- **Key size:** 32 to 448 bits (variable)
- **Rounds:** 16

Blowfish is known for fast encryption after a costly key setup. It uses **key-dependent S-boxes**.

## Structure at a glance
- Split into 32-bit halves (L, R).
- Each round:
  - L = L XOR P[i]
  - R = R XOR F(L)
  - swap (L, R)
- After 16 rounds, swap back and apply final P-entries.

The round function F is built from 4 S-box lookups and arithmetic/XOR mixing.

## Key schedule notes
- Key expansion initializes P-array and S-boxes, then encrypts a fixed block repeatedly to re-key the tables.
- **Key setup is intentionally expensive** (good for some use-cases, bad for frequent re-keying).

## Why it matters for this project
- Demonstrates key-dependent components (S-boxes derived from key material).
- Shows how a Feistel design can use a complex table-driven F-function.

## Practical caveats
- **64-bit block size**: modern high-volume usage can be risky due to birthday-bound effects.
- Key schedule cost makes it less suitable when keys change frequently.

## References
- Bruce Schneier, Blowfish description (academic): https://www.schneier.com/academic/archives/1994/09/description_of_a_new.html
