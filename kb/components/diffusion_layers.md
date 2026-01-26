# Diffusion layers (permutations + linear mixing)

Diffusion spreads the influence of one input bit/byte across many output bits/bytes.

## Permutations
A permutation rearranges bits/bytes:
- **Bit permutations** are common in bit-sliced designs (e.g., Serpent style).
- **Byte permutations** are common in byte-oriented designs (e.g., AES ShiftRows style).

Permutations alone don't create diffusion; they *move* where bits/bytes go so that the next mixing layer spreads them.

## Linear mixing layers
A linear layer combines elements using XOR or finite-field multiplication:
- MixColumns-style diffusion is a standard pattern.
- MDS matrices are popular because they maximize branch number.

## Branch number (intuition)
High branch number means:
- small changes at input force many active S-boxes in subsequent rounds,
  which tends to increase resistance to differential/linear cryptanalysis.

## Practical generator tips
- Ensure the linear layer is **invertible** for SPNs.
- Combine permutation + linear mixing to avoid diffusion being trapped inside small groups.
- Measure diffusion empirically:
  - avalanche by rounds
  - number of active bytes after r rounds (toy proxy)
