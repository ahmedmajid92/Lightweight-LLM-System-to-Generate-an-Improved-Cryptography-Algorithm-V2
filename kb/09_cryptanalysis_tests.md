# Cryptanalysis-oriented tests (lightweight)

These tests **do not prove security**. They are **cheap filters** to detect obviously weak constructions.

## 1) Round-trip correctness (invertibility)
- Encrypt then decrypt random plaintexts.
- Any failure means your construction is broken.

## 2) Avalanche tests (plaintext + key)
Goal (heuristic): flipping 1 input bit changes ~50% of output bits on average.

This is sensitive to:
- number of rounds
- diffusion layer quality
- weak/non-bijective substitution layers

## 3) Bit Independence Criterion (BIC)
BIC checks whether output bits behave independently under input-bit flips.

We approximate BIC by:
- collecting ciphertext difference bit-vectors for many flips
- computing correlation between output bits
- reporting average/max absolute correlation

Lower correlation is better.

## 4) Ciphertext monobit frequency (very lightweight)
Encrypt many random plaintexts under a fixed random key.
Count fraction of `1` bits in ciphertext.
If strongly biased away from ~0.5, the design is suspicious.

## 5) S-box metrics (DDT/LAT)
If you use an 8-bit S-box (AES-like), you can compute:
- **DDT**: maximum differential probability proxy
- **LAT**: maximum linear bias proxy

These are component-level metrics (not full-cipher proofs), but useful.

## What to do when tests fail
- Avalanche too low → increase rounds; strengthen diffusion; replace identity components.
- BIC correlations high → diffusion layer likely weak; avoid linear layers that keep byte positions too independent.
- Monobit bias → S-box or diffusion may be poor; verify key mixing.

## Important
Passing these tests **does NOT** mean secure.
Use them as a research feedback loop, not as a security claim.
