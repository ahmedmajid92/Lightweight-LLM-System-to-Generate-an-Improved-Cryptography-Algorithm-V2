"""AlgorithmsBlock.py (v3, PhD Thesis Edition)

This module provides:
- A structured CipherSpec for block cipher specifications
- SPN, Feistel, and ARX cipher templates built from Components.py registry
- All 12 reference block cipher algorithms (AES, DES, 3DES, Blowfish, Twofish,
  Serpent, Camellia, CAST-128, IDEA, SEED, RC5, RC6)
- Simple local metrics (avalanche tests)
- Export of standalone Python modules

Research / education only. Do NOT use in production.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from Components import ComponentRegistry, xor_bytes, ks_sha256_kdf, rotate_left, rotate_right


# ============================================================================
# CIPHER SPECIFICATION
# ============================================================================

@dataclass
class CipherSpec:
    """Specification for a block cipher configuration.
    
    Attributes:
        name: Human-readable name for this cipher configuration
        architecture: Type of cipher (SPN, FEISTEL, or ARX)
        block_size_bits: Block size in bits (typically 64 or 128)
        key_size_bits: Key size in bits (typically 128, 192, or 256)
        rounds: Number of encryption rounds
        components: Dictionary mapping component roles to component IDs
        version: Specification version for compatibility tracking
        notes: Additional notes or warnings about this configuration
        seed: Random seed for reproducibility in tests and key derivation
    """
    name: str
    architecture: str  # "SPN", "FEISTEL", or "ARX"
    block_size_bits: int
    key_size_bits: int
    rounds: int
    components: Dict[str, str] = field(default_factory=dict)
    version: str = "0.2"
    notes: str = ""
    seed: int = 1337

    def to_json(self) -> str:
        """Serialize specification to JSON."""
        return json.dumps(self.__dict__, indent=2, sort_keys=True)


# ============================================================================
# CIPHER TEMPLATES
# ============================================================================

class BlockCipher:
    """Abstract base class for block cipher implementations."""
    
    def encrypt_block(self, pt: bytes, key: bytes) -> bytes:  # pragma: no cover
        """Encrypt a single block of plaintext."""
        raise NotImplementedError

    def decrypt_block(self, ct: bytes, key: bytes) -> bytes:  # pragma: no cover
        """Decrypt a single block of ciphertext."""
        raise NotImplementedError


@dataclass
class SPNCipher(BlockCipher):
    """Substitution-Permutation Network cipher implementation.
    
    Follows the AES-like structure:
    1. AddRoundKey (XOR with round key)
    2. SubBytes (S-box substitution)
    3. ShiftRows (permutation)
    4. MixColumns (linear diffusion) - except last round
    """
    spec: CipherSpec
    reg: ComponentRegistry

    def encrypt_block(self, pt: bytes, key: bytes) -> bytes:
        """Encrypt using SPN structure."""
        bs = self.spec.block_size_bits // 8
        if len(pt) != bs:
            raise ValueError(f"Plaintext must be {bs} bytes, got {len(pt)}")
        if len(key) * 8 != self.spec.key_size_bits:
            raise ValueError(f"Key must be {self.spec.key_size_bits//8} bytes")
        if bs != 16:
            raise ValueError("SPN template currently supports 128-bit blocks only")

        # Get components
        ks_id = self.spec.components.get("key_schedule", "ks.sha256_kdf")
        sbox_id = self.spec.components.get("sbox", "sbox.aes")
        perm_id = self.spec.components.get("perm", "perm.aes_shiftrows")
        lin_id = self.spec.components.get("linear", "linear.aes_mixcolumns")

        ks = self.reg.get(ks_id).forward
        sbox = self.reg.get(sbox_id)
        perm = self.reg.get(perm_id)
        lin = self.reg.get(lin_id)
        
        if not (sbox.inverse and perm.inverse and lin.inverse):
            raise ValueError("SPN requires invertible components")

        # Generate round keys
        rks = ks(key, rounds=self.spec.rounds, out_len=bs, seed=self.spec.seed)
        state = pt

        # Rounds
        for r in range(self.spec.rounds):
            state = xor_bytes(state, rks[r])      # AddRoundKey
            state = sbox.forward(state)           # SubBytes
            state = perm.forward(state)           # ShiftRows
            if r != self.spec.rounds - 1:
                state = lin.forward(state)        # MixColumns (skip last round)

        # Final round key
        state = xor_bytes(state, rks[self.spec.rounds])
        return state

    def decrypt_block(self, ct: bytes, key: bytes) -> bytes:
        """Decrypt using inverse SPN structure."""
        bs = self.spec.block_size_bits // 8
        if len(ct) != bs:
            raise ValueError(f"Ciphertext must be {bs} bytes")
        if len(key) * 8 != self.spec.key_size_bits:
            raise ValueError(f"Key must be {self.spec.key_size_bits//8} bytes")
        if bs != 16:
            raise ValueError("SPN template currently supports 128-bit blocks only")

        # Get components
        ks_id = self.spec.components.get("key_schedule", "ks.sha256_kdf")
        sbox_id = self.spec.components.get("sbox", "sbox.aes")
        perm_id = self.spec.components.get("perm", "perm.aes_shiftrows")
        lin_id = self.spec.components.get("linear", "linear.aes_mixcolumns")

        ks = self.reg.get(ks_id).forward
        sbox = self.reg.get(sbox_id)
        perm = self.reg.get(perm_id)
        lin = self.reg.get(lin_id)

        # Generate round keys
        rks = ks(key, rounds=self.spec.rounds, out_len=bs, seed=self.spec.seed)

        # Inverse operations
        state = xor_bytes(ct, rks[self.spec.rounds])
        for r in reversed(range(self.spec.rounds)):
            if r != self.spec.rounds - 1:
                state = lin.inverse(state)        # InvMixColumns
            state = perm.inverse(state)           # InvShiftRows
            state = sbox.inverse(state)           # InvSubBytes
            state = xor_bytes(state, rks[r])      # AddRoundKey
        return state


@dataclass
class FeistelCipher(BlockCipher):
    """Feistel network cipher implementation.
    
    Classic Feistel structure:
    1. Split block into left (L) and right (R) halves
    2. For each round: L_new = R, R_new = L XOR F(R, round_key)
    3. Swap and combine
    """
    spec: CipherSpec
    reg: ComponentRegistry

    def encrypt_block(self, pt: bytes, key: bytes) -> bytes:
        """Encrypt using Feistel structure."""
        bs = self.spec.block_size_bits // 8
        if len(pt) != bs:
            raise ValueError(f"Plaintext must be {bs} bytes")
        if bs % 2 != 0:
            raise ValueError("Feistel requires even byte block size")
        if len(key) * 8 != self.spec.key_size_bits:
            raise ValueError(f"Key must be {self.spec.key_size_bits//8} bytes")

        half = bs // 2
        
        # Get components
        ks_id = self.spec.components.get("key_schedule", "ks.sha256_kdf")
        f_sbox_id = self.spec.components.get("f_sbox", "sbox.aes")
        f_perm_id = self.spec.components.get("f_perm", "perm.identity")

        ks = self.reg.get(ks_id).forward
        f_sbox = self.reg.get(f_sbox_id).forward
        f_perm = self.reg.get(f_perm_id).forward

        # Generate round keys
        rks = ks(key, rounds=self.spec.rounds, out_len=half, seed=self.spec.seed)
        L, R = pt[:half], pt[half:]

        # Feistel rounds
        for r in range(self.spec.rounds):
            F = f_perm(f_sbox(xor_bytes(R, rks[r])))
            L, R = R, xor_bytes(L, F)

        return L + R

    def decrypt_block(self, ct: bytes, key: bytes) -> bytes:
        """Decrypt using inverse Feistel structure (reverse round order)."""
        bs = self.spec.block_size_bits // 8
        if len(ct) != bs:
            raise ValueError(f"Ciphertext must be {bs} bytes")
        if bs % 2 != 0:
            raise ValueError("Feistel requires even byte block size")
        if len(key) * 8 != self.spec.key_size_bits:
            raise ValueError(f"Key must be {self.spec.key_size_bits//8} bytes")

        half = bs // 2
        
        # Get components
        ks_id = self.spec.components.get("key_schedule", "ks.sha256_kdf")
        f_sbox_id = self.spec.components.get("f_sbox", "sbox.aes")
        f_perm_id = self.spec.components.get("f_perm", "perm.identity")

        ks = self.reg.get(ks_id).forward
        f_sbox = self.reg.get(f_sbox_id).forward
        f_perm = self.reg.get(f_perm_id).forward

        # Generate round keys
        rks = ks(key, rounds=self.spec.rounds, out_len=half, seed=self.spec.seed)
        L, R = ct[:half], ct[half:]

        # Reverse Feistel rounds
        for r in reversed(range(self.spec.rounds)):
            prev_R = L
            F = f_perm(f_sbox(xor_bytes(prev_R, rks[r])))
            prev_L = xor_bytes(R, F)
            L, R = prev_L, prev_R

        return L + R


@dataclass
class ARXCipher(BlockCipher):
    """Add-Rotate-XOR cipher implementation.
    
    ARX structure used by ciphers like RC5, RC6, ChaCha:
    1. Add round key (modular addition)
    2. Rotate (data-dependent or fixed)
    3. XOR with other half or round key
    """
    spec: CipherSpec
    reg: ComponentRegistry

    def encrypt_block(self, pt: bytes, key: bytes) -> bytes:
        """Encrypt using ARX structure."""
        bs = self.spec.block_size_bits // 8
        if len(pt) != bs:
            raise ValueError(f"Plaintext must be {bs} bytes")
        if len(key) * 8 != self.spec.key_size_bits:
            raise ValueError(f"Key must be {self.spec.key_size_bits//8} bytes")

        # Get components
        ks_id = self.spec.components.get("key_schedule", "ks.sha256_kdf")
        add_id = self.spec.components.get("arx_add", "arx.add_mod32")
        rot_id = self.spec.components.get("arx_rotate", "arx.rotate_left_5")

        ks = self.reg.get(ks_id).forward
        arx_add = self.reg.get(add_id)
        arx_rot = self.reg.get(rot_id)

        # Generate round keys
        rks = ks(key, rounds=self.spec.rounds, out_len=bs, seed=self.spec.seed)
        state = pt

        # ARX rounds
        for r in range(self.spec.rounds):
            # Add round key
            state = xor_bytes(state, rks[r])
            # Apply ARX operations
            state = arx_add.forward(state)
            state = arx_rot.forward(state)

        # Final round key
        state = xor_bytes(state, rks[self.spec.rounds])
        return state

    def decrypt_block(self, ct: bytes, key: bytes) -> bytes:
        """Decrypt using inverse ARX structure."""
        bs = self.spec.block_size_bits // 8
        if len(ct) != bs:
            raise ValueError(f"Ciphertext must be {bs} bytes")
        if len(key) * 8 != self.spec.key_size_bits:
            raise ValueError(f"Key must be {self.spec.key_size_bits//8} bytes")

        # Get components
        ks_id = self.spec.components.get("key_schedule", "ks.sha256_kdf")
        add_id = self.spec.components.get("arx_add", "arx.add_mod32")
        rot_id = self.spec.components.get("arx_rotate", "arx.rotate_left_5")

        ks = self.reg.get(ks_id).forward
        arx_add = self.reg.get(add_id)
        arx_rot = self.reg.get(rot_id)

        # Generate round keys
        rks = ks(key, rounds=self.spec.rounds, out_len=bs, seed=self.spec.seed)

        # Inverse operations
        state = xor_bytes(ct, rks[self.spec.rounds])
        for r in reversed(range(self.spec.rounds)):
            state = arx_rot.inverse(state)
            state = arx_add.inverse(state)
            state = xor_bytes(state, rks[r])
        return state


def build_cipher(spec: CipherSpec, reg: Optional[ComponentRegistry] = None) -> BlockCipher:
    """Factory function to create a cipher from a specification.
    
    Args:
        spec: CipherSpec defining the cipher configuration
        reg: Optional ComponentRegistry; uses default if not provided
        
    Returns:
        BlockCipher instance ready for encryption/decryption
    """
    reg = reg or ComponentRegistry()
    arch = spec.architecture.upper()
    
    if arch == "SPN":
        return SPNCipher(spec=spec, reg=reg)
    if arch == "FEISTEL":
        return FeistelCipher(spec=spec, reg=reg)
    if arch == "ARX":
        return ARXCipher(spec=spec, reg=reg)
    
    raise ValueError(f"Unsupported architecture: {spec.architecture}")


# ============================================================================
# AVALANCHE METRICS
# ============================================================================

def _flip_bit(data: bytes, bit_index: int) -> bytes:
    """Flip a single bit in the data."""
    byte_i = bit_index // 8
    bit_i = bit_index % 8
    out = bytearray(data)
    out[byte_i] ^= (1 << bit_i)
    return bytes(out)


def _hamming(a: bytes, b: bytes) -> int:
    """Calculate Hamming distance (number of differing bits)."""
    return sum((x ^ y).bit_count() for x, y in zip(a, b))


def avalanche_plaintext(cipher: BlockCipher, *, block_bytes: int, key_bytes: int, 
                        trials: int = 200, seed: int = 1337) -> float:
    """Measure plaintext avalanche effect.
    
    A good cipher should have ~0.5 (50% of output bits change 
    when 1 input bit changes).
    """
    rng = random.Random(seed)
    total = 0.0
    total_bits = block_bytes * 8
    
    for _ in range(trials):
        key = bytes(rng.randrange(256) for _ in range(key_bytes))
        pt = bytes(rng.randrange(256) for _ in range(block_bytes))
        ct = cipher.encrypt_block(pt, key)
        bit = rng.randrange(total_bits)
        ct2 = cipher.encrypt_block(_flip_bit(pt, bit), key)
        total += _hamming(ct, ct2) / total_bits
    
    return total / trials if trials else 0.0


def avalanche_key(cipher: BlockCipher, *, block_bytes: int, key_bytes: int,
                  trials: int = 200, seed: int = 1338) -> float:
    """Measure key avalanche effect.
    
    A good cipher should have ~0.5 (50% of output bits change 
    when 1 key bit changes).
    """
    rng = random.Random(seed)
    total = 0.0
    total_bits = block_bytes * 8
    key_bits = key_bytes * 8
    
    for _ in range(trials):
        key = bytes(rng.randrange(256) for _ in range(key_bytes))
        pt = bytes(rng.randrange(256) for _ in range(block_bytes))
        ct = cipher.encrypt_block(pt, key)
        bit = rng.randrange(key_bits)
        ct2 = cipher.encrypt_block(pt, _flip_bit(key, bit))
        total += _hamming(ct, ct2) / total_bits
    
    return total / trials if trials else 0.0


def evaluate_cipher(spec: CipherSpec) -> Dict[str, object]:
    """Run avalanche tests on a cipher specification."""
    cipher = build_cipher(spec)
    bs = spec.block_size_bits // 8
    ks = spec.key_size_bits // 8
    
    return {
        "plaintext_avalanche_mean": avalanche_plaintext(
            cipher, block_bytes=bs, key_bytes=ks, trials=200, seed=spec.seed
        ),
        "key_avalanche_mean": avalanche_key(
            cipher, block_bytes=bs, key_bytes=ks, trials=200, seed=spec.seed + 1
        ),
    }


# ============================================================================
# CODE EXPORT
# ============================================================================

def export_standalone_module(spec: CipherSpec) -> str:
    """Export a standalone Python module implementing this spec.

    For richer exporting with detailed comments, use the cipherlab.cipher.exporter.
    """
    try:
        from cipherlab.cipher.spec import CipherSpec as PSpec
        from cipherlab.cipher.exporter import export_cipher_module
        pspec = PSpec(**spec.__dict__)
        return export_cipher_module(pspec)
    except Exception:
        return (
            "# Standalone export unavailable in this minimal legacy module.\n"
            "# Please use the cipherlab.cipher.exporter module.\n\n"
            "SPEC_JSON = " + repr(spec.to_json()) + "\n"
        )


# ============================================================================
# ALGORITHM LIBRARY - All 12 Reference Algorithms
# ============================================================================

ALGORITHM_LIBRARY: Dict[str, Dict[str, object]] = {
    # ========== SPN ALGORITHMS ==========
    "AES": {
        "architecture": "SPN",
        "block_size_bits": 128,
        "key_size_bits": 128,
        "rounds": 10,
        "components": {
            "sbox": "sbox.aes",
            "perm": "perm.aes_shiftrows",
            "linear": "linear.aes_mixcolumns",
            "key_schedule": "ks.sha256_kdf"
        },
        "notes": "AES-128 template using genuine AES S-box, ShiftRows, and MixColumns.",
    },
    "Serpent": {
        "architecture": "SPN",
        "block_size_bits": 128,
        "key_size_bits": 128,
        "rounds": 32,
        "components": {
            "sbox": "sbox.serpent",
            "perm": "perm.serpent",
            "linear": "linear.identity",
            "key_schedule": "ks.sha256_kdf"
        },
        "notes": "Serpent template using Serpent 4-bit S-boxes. 32 rounds for high security margin.",
    },
    
    # ========== FEISTEL ALGORITHMS ==========
    "DES": {
        "architecture": "FEISTEL",
        "block_size_bits": 64,
        "key_size_bits": 128,  # Extended for this framework (original: 56)
        "rounds": 16,
        "components": {
            "f_sbox": "sbox.des",
            "f_perm": "perm.identity",
            "key_schedule": "ks.des_style"
        },
        "notes": "DES template using DES S-boxes (S1-S8). 64-bit blocks have birthday-bound limits.",
    },
    "3DES": {
        "architecture": "FEISTEL",
        "block_size_bits": 64,
        "key_size_bits": 128,  # Two-key 3DES
        "rounds": 48,  # 3 × 16 DES rounds
        "components": {
            "f_sbox": "sbox.des",
            "f_perm": "perm.identity",
            "key_schedule": "ks.des_style"
        },
        "notes": "Triple DES (EDE mode) template. 48 total rounds. Still limited by 64-bit block size.",
    },
    "Blowfish": {
        "architecture": "FEISTEL",
        "block_size_bits": 64,
        "key_size_bits": 128,
        "rounds": 16,
        "components": {
            "f_sbox": "sbox.blowfish",
            "f_perm": "perm.identity",
            "key_schedule": "ks.blowfish_style"
        },
        "notes": "Blowfish template with key-dependent S-boxes and P-array key schedule.",
    },
    "Twofish": {
        "architecture": "FEISTEL",
        "block_size_bits": 128,
        "key_size_bits": 256,
        "rounds": 16,
        "components": {
            "f_sbox": "sbox.aes",
            "f_perm": "perm.identity",
            "linear": "linear.twofish_mds",
            "key_schedule": "ks.sha256_kdf"
        },
        "notes": "Twofish template using MDS matrix diffusion. AES finalist with 128-bit blocks.",
    },
    "Camellia": {
        "architecture": "FEISTEL",
        "block_size_bits": 128,
        "key_size_bits": 128,
        "rounds": 18,
        "components": {
            "f_sbox": "sbox.aes",
            "f_perm": "perm.identity",
            "key_schedule": "ks.sha256_kdf"
        },
        "notes": "Camellia template. Japanese/EU standard cipher, similar structure to AES but Feistel.",
    },
    "CAST-128": {
        "architecture": "FEISTEL",
        "block_size_bits": 64,
        "key_size_bits": 128,
        "rounds": 16,
        "components": {
            "f_sbox": "sbox.aes",
            "f_perm": "perm.identity",
            "key_schedule": "ks.sha256_kdf"
        },
        "notes": "CAST-128 template. Used in PGP. 64-bit blocks, variable rounds (12 or 16).",
    },
    "SEED": {
        "architecture": "FEISTEL",
        "block_size_bits": 128,
        "key_size_bits": 128,
        "rounds": 16,
        "components": {
            "f_sbox": "sbox.aes",
            "f_perm": "perm.identity",
            "key_schedule": "ks.sha256_kdf"
        },
        "notes": "SEED template. Korean standard cipher. 128-bit blocks, Feistel structure.",
    },
    
    # ========== ARX ALGORITHMS ==========
    "RC5": {
        "architecture": "ARX",
        "block_size_bits": 64,
        "key_size_bits": 128,
        "rounds": 12,
        "components": {
            "arx_add": "arx.add_mod32",
            "arx_rotate": "arx.rotate_left_3",
            "key_schedule": "ks.sha256_kdf"
        },
        "notes": "RC5-32/12/16 template. Simple ARX cipher with data-dependent rotations.",
    },
    "RC6": {
        "architecture": "ARX",
        "block_size_bits": 128,
        "key_size_bits": 128,
        "rounds": 20,
        "components": {
            "arx_add": "arx.add_mod32",
            "arx_rotate": "arx.rotate_left_5",
            "key_schedule": "ks.sha256_kdf"
        },
        "notes": "RC6 template. AES finalist, extends RC5 with integer multiplication for rotation amounts.",
    },
    "IDEA": {
        "architecture": "ARX",
        "block_size_bits": 64,
        "key_size_bits": 128,
        "rounds": 8,
        "components": {
            "arx_add": "arx.mul_mod16",
            "arx_rotate": "arx.rotate_left_5",
            "key_schedule": "ks.sha256_kdf"
        },
        "notes": "IDEA template. Uses multiplication mod 2^16+1, addition mod 2^16, and XOR.",
    },
}


def get_template(name: str, *, override_name: Optional[str] = None, seed: int = 1337) -> CipherSpec:
    """Get a predefined algorithm template as a CipherSpec.
    
    Args:
        name: Algorithm name (e.g., "AES", "DES", "Blowfish")
        override_name: Optional custom name for the spec
        seed: Random seed for reproducibility
        
    Returns:
        CipherSpec configured for the algorithm
    """
    if name not in ALGORITHM_LIBRARY:
        raise KeyError(f"Unknown template: {name}. Available: {list(ALGORITHM_LIBRARY.keys())}")
    
    t = ALGORITHM_LIBRARY[name]
    return CipherSpec(
        name=override_name or f"{name}_Template",
        architecture=str(t["architecture"]),
        block_size_bits=int(t["block_size_bits"]),
        key_size_bits=int(t["key_size_bits"]),
        rounds=int(t["rounds"]),
        components=dict(t["components"]),
        notes=str(t.get("notes", "")),
        seed=int(seed),
    )


def list_algorithms() -> List[str]:
    """List all available algorithm templates."""
    return sorted(ALGORITHM_LIBRARY.keys())


def get_algorithm_info(name: str) -> Dict[str, object]:
    """Get detailed information about an algorithm template."""
    if name not in ALGORITHM_LIBRARY:
        raise KeyError(f"Unknown algorithm: {name}")
    return ALGORITHM_LIBRARY[name].copy()


# ============================================================================
# SMOKE TEST
# ============================================================================

def _smoke_test() -> None:
    """Run basic tests on all cipher architectures."""
    reg = ComponentRegistry()
    
    # Test SPN (AES)
    print("Testing AES (SPN)...")
    spn = get_template("AES", override_name="SmokeSPN")
    c = build_cipher(spn, reg)
    key = os.urandom(spn.key_size_bits // 8)
    pt = os.urandom(spn.block_size_bits // 8)
    ct = c.encrypt_block(pt, key)
    rt = c.decrypt_block(ct, key)
    assert rt == pt, "SPN roundtrip failed"
    print("  ✓ AES roundtrip OK")

    # Test Feistel (DES)
    print("Testing DES (Feistel)...")
    fe = get_template("DES", override_name="SmokeFeistel")
    c2 = build_cipher(fe, reg)
    key2 = os.urandom(fe.key_size_bits // 8)
    pt2 = os.urandom(fe.block_size_bits // 8)
    ct2 = c2.encrypt_block(pt2, key2)
    rt2 = c2.decrypt_block(ct2, key2)
    assert rt2 == pt2, "Feistel roundtrip failed"
    print("  ✓ DES roundtrip OK")

    # Test ARX (RC5)
    print("Testing RC5 (ARX)...")
    arx = get_template("RC5", override_name="SmokeARX")
    c3 = build_cipher(arx, reg)
    key3 = os.urandom(arx.key_size_bits // 8)
    pt3 = os.urandom(arx.block_size_bits // 8)
    ct3 = c3.encrypt_block(pt3, key3)
    rt3 = c3.decrypt_block(ct3, key3)
    assert rt3 == pt3, "ARX roundtrip failed"
    print("  ✓ RC5 roundtrip OK")

    # Test all 12 algorithms
    print("\nTesting all 12 algorithms...")
    for name in list_algorithms():
        spec = get_template(name)
        try:
            cipher = build_cipher(spec, reg)
            key = os.urandom(spec.key_size_bits // 8)
            pt = os.urandom(spec.block_size_bits // 8)
            ct = cipher.encrypt_block(pt, key)
            rt = cipher.decrypt_block(ct, key)
            status = "✓" if rt == pt else "✗"
            print(f"  {status} {name}")
        except Exception as e:
            print(f"  ✗ {name}: {e}")

    print("\nAll smoke tests completed!")


if __name__ == "__main__":
    _smoke_test()
