# HIGHT

**Architecture:** Generalized Feistel (ARX-style F-function)
**Standard:** ISO/IEC 18033-4
**Block size:** 64 bits | **Key size:** 128 bits | **Rounds:** 32

HIGHT (HIGh security and light weigHT) is a lightweight block cipher standardized in ISO/IEC 18033-4. It was designed in South Korea specifically for low-resource devices such as RFID tags and sensor nodes.

HIGHT uses a generalized Feistel structure with an ARX-style round function based on modular addition and rotations. With approximately 3048 gate equivalents, it provides a good balance between security and hardware efficiency. Its 32-round structure with 128-bit keys makes it suitable for IoT applications requiring moderate security levels.
