# Block cipher design principles (high-level)

A **block cipher** is a keyed permutation: for each key *K*, encryption is a bijection on fixed-length blocks.

## Core goals
1. **Correctness / invertibility**
   - Decryption must reverse encryption for all keys and all blocks.
2. **Confusion**
   - Nonlinear components (typically S-boxes) obscure relationships between key/plaintext/ciphertext.
3. **Diffusion**
   - Small changes in plaintext or key should spread across many output bits (avalanche).
4. **Security margin**
   - Enough rounds that known attacks remain infeasible, with margin beyond the best-known attacks.
5. **Implementation constraints**
   - Efficient and safe in the target platform: CPU, embedded, SIMD, low-power, etc.

## Typical structures
- **SPN (Substitution–Permutation Network)**
  - Iterates: key addition + S-box layer + permutation/diffusion layer.
  - AES is a classic example (with linear diffusion in GF(2^8)).
- **Feistel networks**
  - Iterate: swap halves and XOR with a round function.
  - Advantage: invertibility is easier even if round function isn't invertible.

## Common pitfalls in “new cipher” designs
- Weak diffusion layer (low branch number) → differential/linear attacks.
- Poor S-box (low nonlinearity, high differential uniformity) → attacks.
- Weak key schedule → related-key/slid attacks.
- Too few rounds → reduced-round attacks.
- Insecure implementation (timing leakage, table lookups on secret-dependent indices).

## What automated screening can and cannot do
Automated tests can help you reject obviously bad candidates quickly:
- round-trip invertibility
- avalanche heuristics
- S-box DDT/LAT
- basic randomness tests

But automated screening cannot replace deep cryptanalysis.
