# TEA (Tiny Encryption Algorithm)

TEA is one of the simplest practical block ciphers, designed by Wheeler and Needham (1994). Its minimal code size makes it suitable for extremely constrained environments.

## Key Properties
- **Architecture:** Feistel
- **Block size:** 64 bits | **Key size:** 128 bits | **Rounds:** 64 (32 cycles)
- **Operations:** Addition, XOR, shifts only
- **Delta constant:** 0x9E3779B9 (golden ratio derivative)
- **Code size:** Can be implemented in ~10 lines of C

## Known Weaknesses
- Equivalent keys (each key has 3 equivalent keys)
- Related-key attacks
- Addressed by XTEA and XXTEA variants

## Relevance to LWC
TEA demonstrates the minimal complexity needed for a practical Feistel cipher. Its simplicity makes it a reference point for understanding the tradeoff between cipher complexity and security.
