# SIMON (Lightweight Feistel Cipher)

SIMON is a family of lightweight block ciphers designed by the NSA, published in 2013. It is optimized for hardware efficiency (ASIC/FPGA).

## Key Properties
- **Architecture:** Feistel with AND-Rotate-XOR F-function
- **SIMON 64/128:** 64-bit blocks, 128-bit key, 42 rounds
- **F-function:** F(x) = (x <<< 1) AND (x <<< 8) XOR (x <<< 2)
- **Gate count:** ~1751 GE (very low for 64-bit block cipher)

## Relevance to LWC
SIMON is a primary reference for hardware-optimized lightweight ciphers. Its AND-rotate structure provides nonlinearity without S-boxes, making it ideal for constrained ASIC implementations in RFID and IoT.
