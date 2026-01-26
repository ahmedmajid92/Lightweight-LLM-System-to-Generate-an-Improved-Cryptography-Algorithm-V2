# RC5

## High-level summary
- **Architecture:** Parameterized ARX-style cipher (addition, XOR, rotations), often described as Feistel-like in structure
- **Block size:** Variable (commonly 64 bits when word size w=32)
- **Key size:** Variable
- **Rounds:** Variable (commonly 12)

RC5 is notable for **data-dependent rotations**, which makes it different from table-driven S-box ciphers.

## Structure at a glance
RC5 is usually described as RC5-w/r/b:
- w = word size (e.g., 16/32/64)
- r = number of rounds
- b = key length in bytes

It uses:
- XOR
- addition mod 2^w
- rotations where the rotation amount depends on data

## Key schedule notes
- Expands the key into an array of subkeys using “magic constants” and mixing loops.
- Security is sensitive to the chosen parameters (w, r, b).

## Why it matters for this project
- A clean reference for ARX component families (add/xor/rotate).
- Good for exploring how round count and data-dependent rotation impact diffusion.

## Practical caveats
- Parameter choices matter: fewer rounds may be vulnerable to known cryptanalytic techniques on reduced-round variants.
- Correct rotation and word-size handling are essential.

## References
- Rivest, *The RC5 Encryption Algorithm* (MIT): https://people.csail.mit.edu/rivest/pubs/Riv95.pdf
