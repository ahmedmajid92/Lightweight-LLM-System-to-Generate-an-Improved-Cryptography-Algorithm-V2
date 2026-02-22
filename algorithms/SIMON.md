# SIMON

**Architecture:** Feistel (AND-Rotate-XOR)
**Family:** SIMON 64/128
**Block size:** 64 bits | **Key size:** 128 bits | **Rounds:** 42

SIMON is a family of lightweight block ciphers designed by the NSA, optimized for hardware implementations (low gate count, small area). The round function uses bitwise AND, circular rotations, and XOR — no S-boxes or lookup tables.

SIMON 64/128 operates on 64-bit blocks with a 128-bit key over 42 rounds. Its Feistel F-function is defined as: F(x) = (x <<< 1) AND (x <<< 8) XOR (x <<< 2). The simplicity of this function makes SIMON particularly suited for ASIC and FPGA implementations in IoT devices, RFID tags, and sensor networks.
