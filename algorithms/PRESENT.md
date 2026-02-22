# PRESENT

**Architecture:** SPN (Substitution-Permutation Network)
**Standard:** ISO/IEC 29192-2
**Block size:** 64 bits | **Key size:** 80 or 128 bits | **Rounds:** 31

PRESENT is an ultra-lightweight block cipher standardized in ISO/IEC 29192-2 for lightweight cryptography. It uses a 4-bit S-box and a simple bit permutation layer, making it one of the most hardware-efficient SPN ciphers available.

With only ~1570 gate equivalents (GE), PRESENT is designed for extremely constrained environments such as RFID tags and smart cards. The 31-round structure provides adequate security margin despite the small S-box size. Its design influenced subsequent lightweight ciphers including GIFT.
