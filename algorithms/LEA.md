# LEA (Lightweight Encryption Algorithm)

**Architecture:** ARX (Add-Rotate-XOR)
**Standard:** KCSA (Korean Cryptographic Standard Algorithm)
**Block size:** 128 bits | **Key size:** 128 bits | **Rounds:** 24

LEA is a lightweight block cipher designed in South Korea and certified by KCSA. It uses only 32-bit modular addition, rotation, and XOR operations, making it highly efficient on 32-bit software platforms.

LEA processes four 32-bit words per round using different rotation amounts for each word operation. With 128-bit blocks and 128-bit keys over 24 rounds, it provides security comparable to AES while being significantly faster on low-power ARM processors and microcontrollers commonly used in IoT deployments.
