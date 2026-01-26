# Component metadata and interfaces

To build correct ciphers from components, you need a *typed* component library.
Avoid relying on docstring parsing alone.

## Recommended metadata fields
- `cid`: stable unique ID (e.g., `spn.sub.aes_sbox`)
- `kind`: sub / perm / mix / f / key_schedule
- `arch`: SPN / FEISTEL / ANY
- `block_size_bits`: exact size or `None` for any
- `invertible`: boolean
- `inverse_cid`: link to inverse (if invertible)
- `description`: short explanation
- optional: `state_layout`, `unit_size_bits`, `dependencies`

## Recommended function signatures

### SPN state transforms
- substitution: `sub(state: bytes) -> bytes`
- permutation: `perm(state: bytes) -> bytes`
- diffusion: `mix(state: bytes) -> bytes`

### Key schedule
- `key_schedule(master_key: bytes, rounds: int, block_size_bytes: int) -> list[bytes]`

### Feistel round function
- `f(right: bytes, round_key: bytes) -> bytes`

## Compatibility filtering in the UI
When the user picks:
- architecture
- block size

Show only components where:
- `arch` matches (or ANY)
- `block_size_bits` matches (or None)

This prevents invalid constructions at selection time.
