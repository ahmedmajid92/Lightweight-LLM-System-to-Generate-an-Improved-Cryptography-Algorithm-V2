# Block size and birthday-bound limits (64-bit vs 128-bit)

## The birthday bound (why it matters)
For an n-bit block cipher, after about 2^(n/2) blocks, collisions in ciphertext blocks become likely.
This can enable practical attacks in some modes/protocols if huge volumes are encrypted under the same key.

## Practical implication
- 64-bit block ciphers (DES, 3DES, Blowfish, CAST-128, IDEA) can run into volume limits much sooner than 128-bit designs.
- 128-bit block ciphers (AES, Twofish, Serpent, Camellia, SEED, RC6-128) have much larger safety margins for volume.

## Sweet32 (common reference)
“Sweet32” is a well-known practical demonstration of risks when using 64-bit block ciphers in certain settings (e.g., long-lived TLS sessions with CBC).

## Practical advice for this project
- Prefer 128-bit blocks for “modern-style” experiments.
- If you include 64-bit designs, document volume limits and treat them as legacy baselines.

## References
- Sweet32 (NCC Group): https://www.nccgroup.com/us/our-research/technical-advisories/2016/august/technical-advisory-sweet32-birthday-attacks-on-64-bit-block-ciphers-in-tls-and-openvpn/
