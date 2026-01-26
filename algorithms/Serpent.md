# Serpent

## High-level summary
- **Architecture:** Substitution–Permutation Network (SPN), often implemented with **bit-slicing**
- **Block size:** 128 bits
- **Key sizes:** 128 / 192 / 256 bits
- **Rounds:** 32

Serpent was an AES finalist. It is often described as conservative: many rounds, simple and analyzable structure.

## Structure at a glance
Each round uses:
- A 4-bit S-box applied in parallel to the state (bit-sliced friendly)
- A fixed linear transformation for diffusion
- Round key mixing

The last round uses a slightly modified structure (like many SPNs).

## Why it matters for this project
- A strong reference for:
  - Using small S-boxes at scale (4-bit) to create nonlinearity
  - Designing a linear diffusion layer that is easy to implement and reason about
  - Understanding how increasing rounds can buy security margin (at performance cost)

## Practical caveats
- Bit-sliced implementations are fast in software when done carefully, but the representation is different from byte-oriented SPNs.
- A generator that “uses Serpent ideas” should be explicit about:
  - whether it is bit-sliced or byte-oriented
  - how diffusion is achieved and measured

## References
- Serpent proposal paper (Cambridge): https://www.cl.cam.ac.uk/~rja14/Papers/serpent.pdf
- NIST AES project (historical context): https://csrc.nist.gov/projects/aes
