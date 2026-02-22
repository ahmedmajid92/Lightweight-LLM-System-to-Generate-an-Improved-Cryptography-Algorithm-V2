# SPECK

**Architecture:** ARX (Add-Rotate-XOR)
**Family:** SPECK 64/128
**Block size:** 64 bits | **Key size:** 128 bits | **Rounds:** 27

SPECK is a family of lightweight block ciphers designed by the NSA, optimized for software implementations on microcontrollers and embedded processors. It uses only modular addition, bitwise rotation, and XOR.

SPECK 64/128 operates on 64-bit blocks with a 128-bit key over 27 rounds. Each round performs: x = (x >>> alpha) + y XOR k; y = (y <<< beta) XOR x. The lack of lookup tables makes SPECK resistant to cache-timing side-channel attacks and highly efficient on resource-constrained devices.
