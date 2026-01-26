# Evaluation heuristics (what to measure)

This KB section describes fast, automated tests suitable for screening candidate designs.

## 1) Round-trip correctness
- Verify decrypt(encrypt(P,K),K) == P for many random P,K.
- Also check determinism: encrypt(P,K) always returns same C.

## 2) Avalanche effect (heuristic)
Measure average fraction of ciphertext bits that change when:
- flipping one bit in plaintext (plaintext avalanche)
- flipping one bit in key (key avalanche)

For many modern ciphers, a “good” heuristic value is often near 0.5.
But avalanche alone does not guarantee security.

## 3) S-box metrics (if using 8-bit S-box)
- DDT (differential distribution table)
  - max DDT entry → relates to differential uniformity
- LAT (linear approximation table)
  - max absolute correlation → relates to linear bias

These are component-level metrics; full-cipher security requires more.

## 4) Structural red flags
Automated checks can also flag:
- missing key schedule / repeating round keys
- too few rounds
- identity diffusion/permutation
- non-invertible components in SPN

## 5) Extended ideas (thesis-grade)
For stronger evaluation in a thesis:
- reduced-round differential/linear trail search (automatic)
- integral distinguishers for a few rounds
- SAT/SMT-based analysis for small variants
- side-channel risk classification based on component implementation style
