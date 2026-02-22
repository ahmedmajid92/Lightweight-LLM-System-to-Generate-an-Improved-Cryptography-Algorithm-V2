# XTEA (Extended TEA)

**Architecture:** Feistel
**Block size:** 64 bits | **Key size:** 128 bits | **Rounds:** 64 (32 cycles)

XTEA is an improved version of TEA, also designed by Wheeler and Needham. It addresses TEA's key schedule weakness by using a more complex key mixing pattern that prevents related-key attacks.

Like TEA, XTEA uses only addition, XOR, and shifts with a delta constant. The improved key schedule selects different key words per round based on a counter, eliminating the equivalent-key vulnerability. XTEA remains very compact in both hardware and software, making it a practical choice for lightweight IoT applications.
