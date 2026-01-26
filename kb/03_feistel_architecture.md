# Feistel architecture template and constraints

A classic Feistel network splits the block into halves L and R and iterates:

L_{i+1} = R_i  
R_{i+1} = L_i XOR F(R_i, K_i)

Key points:
- The overall mapping is invertible **even if F is not invertible**, as long as XOR and swap are used correctly.
- Decryption applies round keys in reverse.

## Constraints for a Feistel builder
- Block size must be even (split into equal halves).
- F must output exactly the half-block size.
- Keys should be mixed into F (directly or indirectly), otherwise the cipher may have weak key influence.

## Feistel “improvement levers”
- Increase rounds (security margin).
- Strengthen F: include S-box + rotations + mixing with subkey.
- Use better key schedule: avoid repeating subkeys or simple relations between K_i.

## Notes for a component composer
When selecting F components:
- ensure deterministic output length
- avoid non-keyed F unless explicitly exploring such designs
- track whether F uses an S-box and whether it is 4-bit/8-bit etc.
