# S-box design and evaluation (for block ciphers)

## What an S-box does
An S-box is a nonlinear mapping (usually n-bit to n-bit) used to create **confusion**.
In SPNs, it is often the main source of nonlinearity.

## Common “good S-box” properties
For an n-bit bijective S-box (common in block ciphers):
- **Bijective:** every input maps to a unique output (invertible).
- **Low differential uniformity:** limits differential trails.
- **High nonlinearity / low linear bias:** limits linear cryptanalysis.
- **No obvious structural weaknesses:** avoid simple algebraic forms.

## Two standard analysis tools
### Differential Distribution Table (DDT)
For each input difference Δx, count how often output difference Δy occurs:
- lower maximum counts (for Δx ≠ 0) is better.

### Linear Approximation Table (LAT)
For input mask a and output mask b, measure correlation of:
- parity(a·x) == parity(b·S(x))
Lower absolute correlation is better.

## Practical notes for component-based generators
- A random bijection can have decent statistical properties, but can still hide structure.
- Reusing an S-box across rounds is normal, but sometimes designers alternate several S-boxes.
- If you generate S-boxes, keep the generation process reproducible (seeded) and record it.
