# LEA (Lightweight Encryption Algorithm)

LEA is a lightweight block cipher designed in South Korea, certified by KCSA (Korean Cryptographic Standard Algorithm).

## Key Properties
- **Architecture:** ARX (Add-Rotate-XOR)
- **Block size:** 128 bits | **Key size:** 128/192/256 bits | **Rounds:** 24/28/32
- **Word size:** 32-bit operations (four words per block)
- **Operations:** Modular addition, rotation, XOR only
- **Performance:** 1.5-2x faster than AES on ARM Cortex-M

## Relevance to LWC
LEA is the primary 128-bit block ARX cipher in the LWC baseline. Its efficiency on 32-bit ARM processors makes it highly relevant for modern IoT devices. It provides AES-level security with significantly lower power consumption.
