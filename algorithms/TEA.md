# TEA (Tiny Encryption Algorithm)

**Architecture:** Feistel
**Block size:** 64 bits | **Key size:** 128 bits | **Rounds:** 64 (32 cycles)

TEA is one of the simplest block ciphers ever designed, created by David Wheeler and Roger Needham at Cambridge. It uses only addition, XOR, and bit shifts — no S-boxes, no permutation tables.

Each cycle consists of two Feistel rounds using a delta constant (0x9E3779B9, derived from the golden ratio). TEA's extreme simplicity makes it suitable for environments where code size must be minimal. Known weaknesses include related-key attacks and equivalent keys, which led to the development of XTEA.
