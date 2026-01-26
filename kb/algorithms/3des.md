# Triple DES (3DES / TDEA)

## High-level summary
- **Architecture:** DES-based Feistel construction applied 3 times
- **Block size:** 64 bits
- **Keying options:** two-key (112-bit) or three-key (168-bit)
- **Effective rounds:** 3 × 16 DES rounds (48 DES rounds total)

3DES extends DES by applying DES multiple times, typically in **EDE** (Encrypt–Decrypt–Encrypt) form.

## Structure at a glance
For EDE with keys (K1, K2, K3):
- C = E_DES(K3, D_DES(K2, E_DES(K1, P)))

Two-key 3DES usually sets K3 = K1.

## Why it matters for this project
- Illustrates a “construction-based” upgrade: improving key strength without redesigning the primitive.
- Useful for discussing **composition**, keying options, and legacy constraints.

## Practical caveats
- Still a **64-bit block cipher** → birthday-bound issues for large data volumes.
- Higher computational cost than AES and most modern 128-bit block ciphers.
- NIST guidance has progressively restricted new uses of TDEA/3DES.

## References
- NIST SP 800-67 Rev. 2 (TDEA): https://csrc.nist.gov/publications/detail/sp/800-67/rev-2/final
