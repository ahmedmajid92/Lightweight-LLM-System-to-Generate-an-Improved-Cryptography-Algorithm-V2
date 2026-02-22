# SPECK (Lightweight ARX Cipher)

SPECK is a family of lightweight block ciphers designed by the NSA alongside SIMON, published in 2013. It is optimized for software efficiency on microcontrollers.

## Key Properties
- **Architecture:** ARX (Add-Rotate-XOR)
- **SPECK 64/128:** 64-bit blocks, 128-bit key, 27 rounds
- **Round function:** Modular addition, rotation by alpha/beta, XOR
- **No lookup tables:** Resistant to cache-timing attacks

## Relevance to LWC
SPECK is the software counterpart to SIMON. Its pure ARX structure makes it extremely fast on 8/16/32-bit microcontrollers commonly used in IoT devices. It serves as a key reference for software-optimized lightweight cipher design.
