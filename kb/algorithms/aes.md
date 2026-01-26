# AES (Rijndael)

## High-level summary
- **Architecture:** Substitution–Permutation Network (SPN)
- **Block size:** 128 bits
- **Key sizes:** 128 / 192 / 256 bits
- **Rounds:** 10 / 12 / 14 (respectively)

AES is the NIST-standard block cipher and the most common baseline for modern symmetric cryptography.

## Structure at a glance
AES operates on a 4×4 byte **state**. Each round applies:
1. **SubBytes:** a fixed 8-bit S-box applied to each byte (non-linearity)
2. **ShiftRows:** a permutation of bytes (spreads bytes across columns)
3. **MixColumns:** an invertible linear diffusion over GF(2^8)
4. **AddRoundKey:** XOR with a round key

The final round omits MixColumns.

## Key schedule notes
- Expands the master key into **round keys**.
- Uses byte rotations, S-box substitution, and round constants (Rcon) to ensure each key bit influences many round keys.

## Why it matters for this project
AES is a clean reference for:
- How to combine **non-linear confusion** (S-box) + **linear diffusion** (MixColumns) in an SPN.
- How to structure a round so that encryption/decryption are efficient (invertible layers).
- How to evaluate diffusion by round count (avalanche typically improves rapidly after a few rounds).

## Practical caveats
- AES is widely trusted, but **this project’s generated ciphers are not AES**. Borrowing component patterns does not inherit AES’s security.
- If you build an “AES-like” toy SPN, keep it honest: use sufficient rounds and an invertible diffusion layer.

## References
- NIST FIPS 197 (AES): https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.197.pdf
