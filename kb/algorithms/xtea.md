# XTEA (Extended TEA)

XTEA is an improved version of TEA, also designed by Wheeler and Needham (1997). It fixes TEA's key schedule weakness while maintaining simplicity.

## Key Properties
- **Architecture:** Feistel
- **Block size:** 64 bits | **Key size:** 128 bits | **Rounds:** 64 (32 cycles)
- **Key improvement:** Round-dependent key word selection prevents related-key attacks
- **Operations:** Addition, XOR, shifts (same as TEA)

## Relevance to LWC
XTEA provides better security than TEA with minimal additional complexity. It is widely used in embedded systems and IoT applications where code size is critical but TEA's weaknesses are unacceptable.
