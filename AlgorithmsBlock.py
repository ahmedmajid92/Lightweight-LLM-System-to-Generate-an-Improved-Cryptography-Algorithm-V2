"""AlgorithmsBlock.py (v2, OpenAI-ready)

This module provides:
- A structured CipherSpec
- SPN and Feistel cipher templates built from Components.py registry
- Simple local metrics (avalanche tests)
- Export of a standalone Python module for a given spec

Research / education only. Do NOT use in production.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from Components import ComponentRegistry, xor_bytes, ks_sha256_kdf


# ---------- Spec ----------

@dataclass
class CipherSpec:
    name: str
    architecture: str  # "SPN" or "FEISTEL"
    block_size_bits: int
    key_size_bits: int
    rounds: int
    components: Dict[str, str] = field(default_factory=dict)
    version: str = "0.1"
    notes: str = ""
    seed: int = 1337

    def to_json(self) -> str:
        return json.dumps(self.__dict__, indent=2, sort_keys=True)


# ---------- Cipher templates ----------

class BlockCipher:
    def encrypt_block(self, pt: bytes, key: bytes) -> bytes:  # pragma: no cover
        raise NotImplementedError

    def decrypt_block(self, ct: bytes, key: bytes) -> bytes:  # pragma: no cover
        raise NotImplementedError


@dataclass
class SPNCipher(BlockCipher):
    spec: CipherSpec
    reg: ComponentRegistry

    def encrypt_block(self, pt: bytes, key: bytes) -> bytes:
        bs = self.spec.block_size_bits // 8
        if len(pt) != bs:
            raise ValueError(f"pt must be {bs} bytes")
        if len(key) * 8 != self.spec.key_size_bits:
            raise ValueError(f"key must be {self.spec.key_size_bits//8} bytes")
        if bs != 16:
            raise ValueError("SPN template supports 128-bit blocks (16 bytes) only")

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

        rks = ks(key, rounds=self.spec.rounds, out_len=bs, seed=self.spec.seed)
        state = pt

        for r in range(self.spec.rounds):
            state = xor_bytes(state, rks[r])
            state = sbox.forward(state)
            state = perm.forward(state)
            if r != self.spec.rounds - 1:
                state = lin.forward(state)

        state = xor_bytes(state, rks[self.spec.rounds])
        return state

    def decrypt_block(self, ct: bytes, key: bytes) -> bytes:
        bs = self.spec.block_size_bits // 8
        if len(ct) != bs:
            raise ValueError(f"ct must be {bs} bytes")
        if len(key) * 8 != self.spec.key_size_bits:
            raise ValueError(f"key must be {self.spec.key_size_bits//8} bytes")
        if bs != 16:
            raise ValueError("SPN template supports 128-bit blocks (16 bytes) only")

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

        rks = ks(key, rounds=self.spec.rounds, out_len=bs, seed=self.spec.seed)

        state = xor_bytes(ct, rks[self.spec.rounds])
        for r in reversed(range(self.spec.rounds)):
            if r != self.spec.rounds - 1:
                state = lin.inverse(state)
            state = perm.inverse(state)
            state = sbox.inverse(state)
            state = xor_bytes(state, rks[r])
        return state


@dataclass
class FeistelCipher(BlockCipher):
    spec: CipherSpec
    reg: ComponentRegistry

    def encrypt_block(self, pt: bytes, key: bytes) -> bytes:
        bs = self.spec.block_size_bits // 8
        if len(pt) != bs:
            raise ValueError(f"pt must be {bs} bytes")
        if bs % 2 != 0:
            raise ValueError("Feistel requires even byte block size")
        if len(key) * 8 != self.spec.key_size_bits:
            raise ValueError(f"key must be {self.spec.key_size_bits//8} bytes")

        half = bs // 2
        ks_id = self.spec.components.get("key_schedule", "ks.sha256_kdf")
        f_sbox_id = self.spec.components.get("f_sbox", "sbox.aes")
        f_perm_id = self.spec.components.get("f_perm", "perm.identity")

        ks = self.reg.get(ks_id).forward
        f_sbox = self.reg.get(f_sbox_id).forward
        f_perm = self.reg.get(f_perm_id).forward

        rks = ks(key, rounds=self.spec.rounds, out_len=half, seed=self.spec.seed)
        L, R = pt[:half], pt[half:]

        for r in range(self.spec.rounds):
            F = f_perm(f_sbox(xor_bytes(R, rks[r])))
            L, R = R, xor_bytes(L, F)

        return L + R

    def decrypt_block(self, ct: bytes, key: bytes) -> bytes:
        bs = self.spec.block_size_bits // 8
        if len(ct) != bs:
            raise ValueError(f"ct must be {bs} bytes")
        if bs % 2 != 0:
            raise ValueError("Feistel requires even byte block size")
        if len(key) * 8 != self.spec.key_size_bits:
            raise ValueError(f"key must be {self.spec.key_size_bits//8} bytes")

        half = bs // 2
        ks_id = self.spec.components.get("key_schedule", "ks.sha256_kdf")
        f_sbox_id = self.spec.components.get("f_sbox", "sbox.aes")
        f_perm_id = self.spec.components.get("f_perm", "perm.identity")

        ks = self.reg.get(ks_id).forward
        f_sbox = self.reg.get(f_sbox_id).forward
        f_perm = self.reg.get(f_perm_id).forward

        rks = ks(key, rounds=self.spec.rounds, out_len=half, seed=self.spec.seed)
        L, R = ct[:half], ct[half:]

        for r in reversed(range(self.spec.rounds)):
            prev_R = L
            F = f_perm(f_sbox(xor_bytes(prev_R, rks[r])))
            prev_L = xor_bytes(R, F)
            L, R = prev_L, prev_R

        return L + R


def build_cipher(spec: CipherSpec, reg: Optional[ComponentRegistry] = None) -> BlockCipher:
    reg = reg or ComponentRegistry()
    arch = spec.architecture.upper()
    if arch == "SPN":
        return SPNCipher(spec=spec, reg=reg)
    if arch == "FEISTEL":
        return FeistelCipher(spec=spec, reg=reg)
    raise ValueError(f"Unsupported architecture: {spec.architecture}")


# ---------- Metrics (simple avalanche tests) ----------

def _flip_bit(data: bytes, bit_index: int) -> bytes:
    byte_i = bit_index // 8
    bit_i = bit_index % 8
    out = bytearray(data)
    out[byte_i] ^= (1 << bit_i)
    return bytes(out)


def _hamming(a: bytes, b: bytes) -> int:
    return sum((x ^ y).bit_count() for x, y in zip(a, b))


def avalanche_plaintext(cipher: BlockCipher, *, block_bytes: int, key_bytes: int, trials: int = 200, seed: int = 1337) -> float:
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


def avalanche_key(cipher: BlockCipher, *, block_bytes: int, key_bytes: int, trials: int = 200, seed: int = 1338) -> float:
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
    cipher = build_cipher(spec)
    bs = spec.block_size_bits // 8
    ks = spec.key_size_bits // 8
    return {
        "plaintext_avalanche_mean": avalanche_plaintext(cipher, block_bytes=bs, key_bytes=ks, trials=200, seed=spec.seed),
        "key_avalanche_mean": avalanche_key(cipher, block_bytes=bs, key_bytes=ks, trials=200, seed=spec.seed + 1),
    }


# ---------- Export (standalone module) ----------

def export_standalone_module(spec: CipherSpec) -> str:
    """Export a standalone Python module implementing this spec.

    For richer exporting (selective component inclusion, run tracking),
    prefer the `cipherlab.cipher.exporter` in the OpenAI v2 project.
    """
    # Minimal exporter delegates to the v2 project's exporter if available.
    try:
        from cipherlab.cipher.spec import CipherSpec as PSpec
        from cipherlab.cipher.exporter import export_cipher_module
        pspec = PSpec(**spec.__dict__)
        return export_cipher_module(pspec)
    except Exception:
        # Fallback: serialize spec + instruct user to use the v2 project exporter
        return (
            "# Standalone export unavailable in this minimal legacy module.\n"
            "# Please use the OpenAI v2 project exporter.\n\n"
            "SPEC_JSON = " + repr(spec.to_json()) + "\n"
        )


# ---------- Algorithm templates (metadata, not exact implementations) ----------

ALGORITHM_LIBRARY: Dict[str, Dict[str, object]] = {
    "AES": {
        "architecture": "SPN",
        "block_size_bits": 128,
        "key_size_bits": 128,
        "rounds": 10,
        "components": {"sbox": "sbox.aes", "perm": "perm.aes_shiftrows", "linear": "linear.aes_mixcolumns", "key_schedule": "ks.sha256_kdf"},
        "notes": "AES-like SPN template using AES S-box/ShiftRows/MixColumns, but NOT the AES key schedule.",
    },
    "DES": {
        "architecture": "FEISTEL",
        "block_size_bits": 64,
        "key_size_bits": 128,
        "rounds": 16,
        "components": {"f_sbox": "sbox.aes", "f_perm": "perm.identity", "key_schedule": "ks.sha256_kdf"},
        "notes": "Feistel template (DES-inspired form factor). Not a faithful DES implementation.",
    },
    "Twofish": {
        "architecture": "FEISTEL",
        "block_size_bits": 128,
        "key_size_bits": 256,
        "rounds": 16,
        "components": {"f_sbox": "sbox.aes", "f_perm": "perm.identity", "key_schedule": "ks.sha256_kdf"},
        "notes": "Feistel template matching Twofish block/round counts, but not a faithful Twofish implementation.",
    },
}


def get_template(name: str, *, override_name: Optional[str] = None, seed: int = 1337) -> CipherSpec:
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


# ---------- Quick smoke test ----------

def _smoke_test() -> None:
    reg = ComponentRegistry()
    spn = get_template("AES", override_name="SmokeSPN")
    c = build_cipher(spn, reg)
    key = os.urandom(spn.key_size_bits // 8)
    pt = os.urandom(spn.block_size_bits // 8)
    ct = c.encrypt_block(pt, key)
    rt = c.decrypt_block(ct, key)
    assert rt == pt

    fe = get_template("DES", override_name="SmokeFeistel")
    c2 = build_cipher(fe, reg)
    key2 = os.urandom(fe.key_size_bits // 8)
    pt2 = os.urandom(fe.block_size_bits // 8)
    ct2 = c2.encrypt_block(pt2, key2)
    rt2 = c2.decrypt_block(ct2, key2)
    assert rt2 == pt2

if __name__ == "__main__":
    _smoke_test()
    print("Smoke test OK")
