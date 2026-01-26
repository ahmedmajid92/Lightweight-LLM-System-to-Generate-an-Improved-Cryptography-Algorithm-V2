# Avalanche tests (block ciphers)

## Avalanche effect (intuition)
A block cipher should exhibit strong diffusion:
- flipping 1 bit of plaintext should flip about **half** the ciphertext bits on average.
- similarly for flipping 1 bit of the key (key avalanche).

## Strict Avalanche Criterion (SAC)
A stronger form:
- each output bit should change with probability 1/2 when a single input bit is complemented.

## Why avalanche is not enough
A cipher can have good avalanche statistics and still be insecure.
Avalanche tests are useful as *early feedback* for toy designs, not as proof of security.

## Practical test recipe
1. Choose random key and plaintext.
2. Encrypt -> C.
3. Flip one bit in plaintext -> P'.
4. Encrypt -> C'.
5. Measure Hamming distance between C and C'. Repeat many times.
6. Compute mean fraction of bits changed.

Repeat for key bits (fix plaintext, flip key bits).
