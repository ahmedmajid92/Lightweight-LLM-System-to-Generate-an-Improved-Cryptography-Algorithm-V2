# GIFT

**Architecture:** SPN (Substitution-Permutation Network)
**Block size:** 128 bits | **Key size:** 128 bits | **Rounds:** 40

GIFT is a lightweight block cipher that improves upon PRESENT's design. It comes in two variants: GIFT-64 (64-bit blocks) and GIFT-128 (128-bit blocks). GIFT-128 is the more commonly used variant.

GIFT uses a 4-bit S-box followed by a bit permutation layer, similar to PRESENT but with a more efficient permutation design that provides better diffusion with fewer rounds of overhead. The cipher achieves strong security with approximately 1700 GE (for GIFT-64), making it competitive with PRESENT while providing improved resistance to linear and differential cryptanalysis.
