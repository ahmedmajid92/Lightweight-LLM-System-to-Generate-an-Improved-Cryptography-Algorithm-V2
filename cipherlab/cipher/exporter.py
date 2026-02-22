from __future__ import annotations

from typing import Dict, List

from .spec import CipherSpec


def export_cipher_module(spec: CipherSpec) -> str:
    """Return a standalone Python module implementing the cipher spec."""

    # Decide what code pieces to include
    uses_aes_sbox = any(v == "sbox.aes" for v in spec.components.values())
    uses_present_sbox = any(v == "sbox.present" for v in spec.components.values())
    uses_gift_sbox = any(v == "sbox.gift" for v in spec.components.values())
    uses_shiftrows = any(v == "perm.aes_shiftrows" for v in spec.components.values())
    uses_present_perm = any(v == "perm.present" for v in spec.components.values())
    uses_gift_perm = any(v == "perm.gift" for v in spec.components.values())
    uses_mixcolumns = any(v == "linear.aes_mixcolumns" for v in spec.components.values())
    uses_tea_f = any(v == "sbox.tea_f" for v in spec.components.values())
    uses_xtea_f = any(v == "sbox.xtea_f" for v in spec.components.values())
    uses_simon_f = any(v == "sbox.simon_f" for v in spec.components.values())
    uses_hight_f = any(v == "sbox.hight_f" for v in spec.components.values())

    header = f'''"""Generated block cipher (research prototype)

Name: {spec.name}
Architecture: {spec.architecture}
Block size: {spec.block_size_bits} bits
Key size: {spec.key_size_bits} bits
Rounds: {spec.rounds}

WARNING:
- This is for research/education only.
- Do NOT use in production.
"""

'''

    util = """from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from typing import List

def xor_bytes(a: bytes, b: bytes) -> bytes:
    if len(a) != len(b):
        raise ValueError("xor_bytes length mismatch")
    return bytes(x ^ y for x, y in zip(a, b))

def rotate_left(x: int, r: int, w: int) -> int:
    mask = (1 << w) - 1
    r &= (w - 1)
    x &= mask
    return ((x << r) & mask) | (x >> (w - r))

def rotate_right(x: int, r: int, w: int) -> int:
    mask = (1 << w) - 1
    r &= (w - 1)
    x &= mask
    return (x >> r) | ((x << (w - r)) & mask)

def bytes_to_words(data: bytes, word_size: int = 4, byteorder: str = 'little') -> List[int]:
    fmt = '<' if byteorder == 'little' else '>'
    fmt += {2: 'H', 4: 'I', 8: 'Q'}[word_size] * (len(data) // word_size)
    return list(struct.unpack(fmt, data))

def words_to_bytes(words: List[int], word_size: int = 4, byteorder: str = 'little') -> bytes:
    fmt = '<' if byteorder == 'little' else '>'
    fmt += {2: 'H', 4: 'I', 8: 'Q'}[word_size] * len(words)
    return struct.pack(fmt, *words)

def _sha256(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()

def ks_sha256_kdf(key: bytes, *, rounds: int, out_len: int, seed: int = 0) -> List[bytes]:
    seed_b = int(seed).to_bytes(4, "little", signed=False)
    keys: List[bytes] = []
    for r in range(rounds + 1):
        r_b = int(r).to_bytes(4, "little", signed=False)
        base = _sha256(key + seed_b + r_b)
        buf = b"""
    util += """\
"
        ctr = 0
        while len(buf) < out_len:
            buf += _sha256(base + int(ctr).to_bytes(4, "little", signed=False))
            ctr += 1
        keys.append(buf[:out_len])
    return keys
"""

    # ---- S-box code ----
    sbox_code = ""
    if uses_aes_sbox:
        sbox_code += """AES_SBOX = [
    0x63,0x7c,0x77,0x7b,0xf2,0x6b,0x6f,0xc5,0x30,0x01,0x67,0x2b,0xfe,0xd7,0xab,0x76,
    0xca,0x82,0xc9,0x7d,0xfa,0x59,0x47,0xf0,0xad,0xd4,0xa2,0xaf,0x9c,0xa4,0x72,0xc0,
    0xb7,0xfd,0x93,0x26,0x36,0x3f,0xf7,0xcc,0x34,0xa5,0xe5,0xf1,0x71,0xd8,0x31,0x15,
    0x04,0xc7,0x23,0xc3,0x18,0x96,0x05,0x9a,0x07,0x12,0x80,0xe2,0xeb,0x27,0xb2,0x75,
    0x09,0x83,0x2c,0x1a,0x1b,0x6e,0x5a,0xa0,0x52,0x3b,0xd6,0xb3,0x29,0xe3,0x2f,0x84,
    0x53,0xd1,0x00,0xed,0x20,0xfc,0xb1,0x5b,0x6a,0xcb,0xbe,0x39,0x4a,0x4c,0x58,0xcf,
    0xd0,0xef,0xaa,0xfb,0x43,0x4d,0x33,0x85,0x45,0xf9,0x02,0x7f,0x50,0x3c,0x9f,0xa8,
    0x51,0xa3,0x40,0x8f,0x92,0x9d,0x38,0xf5,0xbc,0xb6,0xda,0x21,0x10,0xff,0xf3,0xd2,
    0xcd,0x0c,0x13,0xec,0x5f,0x97,0x44,0x17,0xc4,0xa7,0x7e,0x3d,0x64,0x5d,0x19,0x73,
    0x60,0x81,0x4f,0xdc,0x22,0x2a,0x90,0x88,0x46,0xee,0xb8,0x14,0xde,0x5e,0x0b,0xdb,
    0xe0,0x32,0x3a,0x0a,0x49,0x06,0x24,0x5c,0xc2,0xd3,0xac,0x62,0x91,0x95,0xe4,0x79,
    0xe7,0xc8,0x37,0x6d,0x8d,0xd5,0x4e,0xa9,0x6c,0x56,0xf4,0xea,0x65,0x7a,0xae,0x08,
    0xba,0x78,0x25,0x2e,0x1c,0xa6,0xb4,0xc6,0xe8,0xdd,0x74,0x1f,0x4b,0xbd,0x8b,0x8a,
    0x70,0x3e,0xb5,0x66,0x48,0x03,0xf6,0x0e,0x61,0x35,0x57,0xb9,0x86,0xc1,0x1d,0x9e,
    0xe1,0xf8,0x98,0x11,0x69,0xd9,0x8e,0x94,0x9b,0x1e,0x87,0xe9,0xce,0x55,0x28,0xdf,
    0x8c,0xa1,0x89,0x0d,0xbf,0xe6,0x42,0x68,0x41,0x99,0x2d,0x0f,0xb0,0x54,0xbb,0x16,
]
AES_INV_SBOX = [0] * 256
for i, v in enumerate(AES_SBOX):
    AES_INV_SBOX[v] = i

def sbox_aes(data: bytes) -> bytes:
    return bytes(AES_SBOX[b] for b in data)

def sboxinv_aes(data: bytes) -> bytes:
    return bytes(AES_INV_SBOX[b] for b in data)
"""

    if uses_present_sbox:
        sbox_code += """PRESENT_SBOX = [0xC, 0x5, 0x6, 0xB, 0x9, 0x0, 0xA, 0xD,
                0x3, 0xE, 0xF, 0x8, 0x4, 0x7, 0x1, 0x2]
PRESENT_INV_SBOX = [0] * 16
for i, v in enumerate(PRESENT_SBOX):
    PRESENT_INV_SBOX[v] = i

def sbox_present(data: bytes) -> bytes:
    result = bytearray(len(data))
    for i, byte in enumerate(data):
        lo = PRESENT_SBOX[byte & 0x0F]
        hi = PRESENT_SBOX[(byte >> 4) & 0x0F]
        result[i] = (hi << 4) | lo
    return bytes(result)

def sboxinv_present(data: bytes) -> bytes:
    result = bytearray(len(data))
    for i, byte in enumerate(data):
        lo = PRESENT_INV_SBOX[byte & 0x0F]
        hi = PRESENT_INV_SBOX[(byte >> 4) & 0x0F]
        result[i] = (hi << 4) | lo
    return bytes(result)
"""

    if uses_gift_sbox:
        sbox_code += """GIFT_SBOX = [0x1, 0xA, 0x4, 0xC, 0x6, 0xF, 0x3, 0x9,
             0x2, 0xD, 0xB, 0x7, 0x5, 0x0, 0x8, 0xE]
GIFT_INV_SBOX = [0] * 16
for i, v in enumerate(GIFT_SBOX):
    GIFT_INV_SBOX[v] = i

def sbox_gift(data: bytes) -> bytes:
    result = bytearray(len(data))
    for i, byte in enumerate(data):
        lo = GIFT_SBOX[byte & 0x0F]
        hi = GIFT_SBOX[(byte >> 4) & 0x0F]
        result[i] = (hi << 4) | lo
    return bytes(result)

def sboxinv_gift(data: bytes) -> bytes:
    result = bytearray(len(data))
    for i, byte in enumerate(data):
        lo = GIFT_INV_SBOX[byte & 0x0F]
        hi = GIFT_INV_SBOX[(byte >> 4) & 0x0F]
        result[i] = (hi << 4) | lo
    return bytes(result)
"""

    # ---- Feistel F-function code ----
    feistel_f_code = ""
    if uses_tea_f:
        feistel_f_code += """def sbox_tea_f(data: bytes) -> bytes:
    words = bytes_to_words(data, 4, 'little')
    result = []
    for w in words:
        shifted = (((w << 4) & 0xFFFFFFFF) ^ ((w >> 5) & 0xFFFFFFFF))
        result.append((shifted + w) & 0xFFFFFFFF)
    return words_to_bytes(result, 4, 'little')
"""
    if uses_xtea_f:
        feistel_f_code += """def sbox_xtea_f(data: bytes) -> bytes:
    words = bytes_to_words(data, 4, 'little')
    result = []
    for w in words:
        shifted = (((w << 4) & 0xFFFFFFFF) ^ ((w >> 5) & 0xFFFFFFFF))
        result.append((shifted + w) & 0xFFFFFFFF)
    return words_to_bytes(result, 4, 'little')
"""
    if uses_simon_f:
        feistel_f_code += """def sbox_simon_f(data: bytes) -> bytes:
    words = bytes_to_words(data, 4, 'little')
    result = []
    for x in words:
        r1 = rotate_left(x, 1, 32)
        r8 = rotate_left(x, 8, 32)
        r2 = rotate_left(x, 2, 32)
        result.append((r1 & r8) ^ r2)
    return words_to_bytes(result, 4, 'little')
"""
    if uses_hight_f:
        feistel_f_code += """def _hight_f0(x: int) -> int:
    return rotate_left(x, 1, 8) ^ rotate_left(x, 2, 8) ^ rotate_left(x, 7, 8)

def _hight_f1(x: int) -> int:
    return rotate_left(x, 3, 8) ^ rotate_left(x, 4, 8) ^ rotate_left(x, 6, 8)

def sbox_hight_f(data: bytes) -> bytes:
    result = bytearray(len(data))
    for i, b in enumerate(data):
        result[i] = _hight_f0(b) if i % 2 == 0 else _hight_f1(b)
    return bytes(result)
"""

    # ---- Permutation code ----
    perm_code = ""
    if uses_shiftrows:
        perm_code += """# AES ShiftRows in column-major layout (index = col*4 + row)
_AES_SHIFTROWS_MAP = []
_AES_INV_SHIFTROWS_MAP = []
for col in range(4):
    for row in range(4):
        old_col = (col + row) % 4
        _AES_SHIFTROWS_MAP.append(old_col * 4 + row)
for col in range(4):
    for row in range(4):
        old_col = (col - row) % 4
        _AES_INV_SHIFTROWS_MAP.append(old_col * 4 + row)

def perm_aes_shiftrows(data: bytes) -> bytes:
    if len(data) != 16:
        raise ValueError("perm_aes_shiftrows requires 16-byte blocks")
    return bytes(data[i] for i in _AES_SHIFTROWS_MAP)

def perm_aes_inv_shiftrows(data: bytes) -> bytes:
    if len(data) != 16:
        raise ValueError("perm_aes_inv_shiftrows requires 16-byte blocks")
    return bytes(data[i] for i in _AES_INV_SHIFTROWS_MAP)
"""

    if uses_present_perm:
        perm_code += """PRESENT_PERM = [0] * 64
for _i in range(64):
    PRESENT_PERM[_i] = (_i * 16) % 63 if _i < 63 else 63
PRESENT_INV_PERM = [0] * 64
for _i in range(64):
    PRESENT_INV_PERM[PRESENT_PERM[_i]] = _i

def perm_present(data: bytes) -> bytes:
    if len(data) != 8:
        raise ValueError("perm_present requires 8-byte (64-bit) input")
    val = int.from_bytes(data, 'big')
    out = 0
    for i in range(64):
        if val & (1 << (63 - i)):
            out |= (1 << (63 - PRESENT_PERM[i]))
    return out.to_bytes(8, 'big')

def perm_present_inv(data: bytes) -> bytes:
    if len(data) != 8:
        raise ValueError("perm_present_inv requires 8-byte (64-bit) input")
    val = int.from_bytes(data, 'big')
    out = 0
    for i in range(64):
        if val & (1 << (63 - i)):
            out |= (1 << (63 - PRESENT_INV_PERM[i]))
    return out.to_bytes(8, 'big')
"""

    if uses_gift_perm:
        perm_code += """GIFT_PERM_128 = [
     0,  33,  66,  99,   2,  35,  68, 101,   4,  37,  70, 103,   6,  39,  72, 105,
     8,  41,  74, 107,  10,  43,  76, 109,  12,  45,  78, 111,  14,  47,  80, 113,
    16,  49,  82, 115,  18,  51,  84, 117,  20,  53,  86, 119,  22,  55,  88, 121,
    24,  57,  90, 123,  26,  59,  92, 125,  28,  61,  94, 127,  30,  63,  64,  97,
    32,   1,  98,  67,  34,   3, 100,  69,  36,   5, 102,  71,  38,   7, 104,  73,
    40,   9, 106,  75,  42,  11, 108,  77,  44,  13, 110,  79,  46,  15, 112,  81,
    48,  17, 114,  83,  50,  19, 116,  85,  52,  21, 118,  87,  54,  23, 120,  89,
    56,  25, 122,  91,  58,  27, 124,  93,  60,  29, 126,  95,  62,  31,  96,  65,
]
GIFT_INV_PERM_128 = [0] * 128
for _i, _v in enumerate(GIFT_PERM_128):
    GIFT_INV_PERM_128[_v] = _i

def perm_gift(data: bytes) -> bytes:
    if len(data) != 16:
        raise ValueError("perm_gift requires 16-byte (128-bit) input")
    val = int.from_bytes(data, 'big')
    out = 0
    for i in range(128):
        if val & (1 << (127 - i)):
            out |= (1 << (127 - GIFT_PERM_128[i]))
    return out.to_bytes(16, 'big')

def perm_gift_inv(data: bytes) -> bytes:
    if len(data) != 16:
        raise ValueError("perm_gift_inv requires 16-byte (128-bit) input")
    val = int.from_bytes(data, 'big')
    out = 0
    for i in range(128):
        if val & (1 << (127 - i)):
            out |= (1 << (127 - GIFT_INV_PERM_128[i]))
    return out.to_bytes(16, 'big')
"""

    # ---- Linear layer code ----
    linear_code = ""
    if uses_mixcolumns:
        linear_code = """def _mul(a: int, b: int) -> int:
    a &= 0xFF
    b &= 0xFF
    res = 0
    for _ in range(8):
        if b & 1:
            res ^= a
        hi = a & 0x80
        a = (a << 1) & 0xFF
        if hi:
            a ^= 0x1B
        b >>= 1
    return res & 0xFF

def linear_aes_mixcolumns(data: bytes) -> bytes:
    if len(data) != 16:
        raise ValueError("linear_aes_mixcolumns requires 16-byte blocks")
    out = bytearray(16)
    for c in range(4):
        i = c * 4
        a0, a1, a2, a3 = data[i], data[i+1], data[i+2], data[i+3]
        out[i+0] = _mul(a0,2) ^ _mul(a1,3) ^ a2 ^ a3
        out[i+1] = a0 ^ _mul(a1,2) ^ _mul(a2,3) ^ a3
        out[i+2] = a0 ^ a1 ^ _mul(a2,2) ^ _mul(a3,3)
        out[i+3] = _mul(a0,3) ^ a1 ^ a2 ^ _mul(a3,2)
    return bytes(out)

def linear_aes_inv_mixcolumns(data: bytes) -> bytes:
    if len(data) != 16:
        raise ValueError("linear_aes_inv_mixcolumns requires 16-byte blocks")
    out = bytearray(16)
    for c in range(4):
        i = c * 4
        a0, a1, a2, a3 = data[i], data[i+1], data[i+2], data[i+3]
        out[i+0] = _mul(a0,14) ^ _mul(a1,11) ^ _mul(a2,13) ^ _mul(a3,9)
        out[i+1] = _mul(a0,9) ^ _mul(a1,14) ^ _mul(a2,11) ^ _mul(a3,13)
        out[i+2] = _mul(a0,13) ^ _mul(a1,9) ^ _mul(a2,14) ^ _mul(a3,11)
        out[i+3] = _mul(a0,11) ^ _mul(a1,13) ^ _mul(a2,9) ^ _mul(a3,14)
    return bytes(out)
"""

    # ---- Architecture-specific cipher class ----
    def _resolve_sbox(comp_id):
        """Map component ID to function name in exported module."""
        mapping = {
            "sbox.aes": ("sbox_aes", "sboxinv_aes"),
            "sbox.present": ("sbox_present", "sboxinv_present"),
            "sbox.gift": ("sbox_gift", "sboxinv_gift"),
            "sbox.tea_f": ("sbox_tea_f", "sbox_tea_f"),
            "sbox.xtea_f": ("sbox_xtea_f", "sbox_xtea_f"),
            "sbox.simon_f": ("sbox_simon_f", "sbox_simon_f"),
            "sbox.hight_f": ("sbox_hight_f", "sbox_hight_f"),
        }
        return mapping.get(comp_id, ("(lambda x: x)", "(lambda x: x)"))

    def _resolve_perm(comp_id):
        mapping = {
            "perm.aes_shiftrows": ("perm_aes_shiftrows", "perm_aes_inv_shiftrows"),
            "perm.present": ("perm_present", "perm_present_inv"),
            "perm.gift": ("perm_gift", "perm_gift_inv"),
            "perm.identity": ("(lambda x: x)", "(lambda x: x)"),
        }
        return mapping.get(comp_id, ("(lambda x: x)", "(lambda x: x)"))

    def _resolve_linear(comp_id):
        mapping = {
            "linear.aes_mixcolumns": ("linear_aes_mixcolumns", "linear_aes_inv_mixcolumns"),
            "linear.identity": ("(lambda x: x)", "(lambda x: x)"),
        }
        return mapping.get(comp_id, ("(lambda x: x)", "(lambda x: x)"))

    if spec.architecture == "SPN":
        sbox_f, sbox_i = _resolve_sbox(spec.components.get("sbox", "sbox.aes"))
        perm_f, perm_i = _resolve_perm(spec.components.get("perm", "perm.aes_shiftrows"))
        lin_f, lin_i = _resolve_linear(spec.components.get("linear", "linear.aes_mixcolumns"))

        arch_code = f"""@dataclass
class Cipher:
    rounds: int = {spec.rounds}
    block_size_bytes: int = {spec.block_size_bits // 8}
    key_size_bytes: int = {spec.key_size_bits // 8}
    seed: int = {spec.seed}

    def encrypt_block(self, pt: bytes, key: bytes) -> bytes:
        if len(pt) != self.block_size_bytes:
            raise ValueError(\"bad pt length\")
        if len(key) != self.key_size_bytes:
            raise ValueError(\"bad key length\")
        rks = ks_sha256_kdf(key, rounds=self.rounds, out_len=self.block_size_bytes, seed=self.seed)
        state = pt
        for r in range(self.rounds):
            state = xor_bytes(state, rks[r])
            state = {sbox_f}(state)
            state = {perm_f}(state)
            if r != self.rounds - 1:
                state = {lin_f}(state)
        state = xor_bytes(state, rks[self.rounds])
        return state

    def decrypt_block(self, ct: bytes, key: bytes) -> bytes:
        if len(ct) != self.block_size_bytes:
            raise ValueError(\"bad ct length\")
        if len(key) != self.key_size_bytes:
            raise ValueError(\"bad key length\")
        rks = ks_sha256_kdf(key, rounds=self.rounds, out_len=self.block_size_bytes, seed=self.seed)
        state = xor_bytes(ct, rks[self.rounds])
        for r in reversed(range(self.rounds)):
            if r != self.rounds - 1:
                state = {lin_i}(state)
            state = {perm_i}(state)
            state = {sbox_i}(state)
            state = xor_bytes(state, rks[r])
        return state

def self_test() -> None:
    import os
    c = Cipher()
    for _ in range(50):
        key = os.urandom(c.key_size_bytes)
        pt = os.urandom(c.block_size_bytes)
        ct = c.encrypt_block(pt, key)
        rt = c.decrypt_block(ct, key)
        assert rt == pt, (pt.hex(), ct.hex(), rt.hex())
    print(\"self_test OK\")
"""
    elif spec.architecture == "FEISTEL":
        sbox_f, _ = _resolve_sbox(spec.components.get("f_sbox", "sbox.aes"))
        perm_f, _ = _resolve_perm(spec.components.get("f_perm", "perm.identity"))

        arch_code = f"""@dataclass
class Cipher:
    rounds: int = {spec.rounds}
    block_size_bytes: int = {spec.block_size_bits // 8}
    key_size_bytes: int = {spec.key_size_bits // 8}
    seed: int = {spec.seed}

    def encrypt_block(self, pt: bytes, key: bytes) -> bytes:
        if len(pt) != self.block_size_bytes:
            raise ValueError(\"bad pt length\")
        if len(key) != self.key_size_bytes:
            raise ValueError(\"bad key length\")
        if self.block_size_bytes % 2 != 0:
            raise ValueError(\"Feistel needs even byte blocks\")
        half = self.block_size_bytes // 2
        rks = ks_sha256_kdf(key, rounds=self.rounds, out_len=half, seed=self.seed)
        L, R = pt[:half], pt[half:]
        for r in range(self.rounds):
            F = {perm_f}({sbox_f}(xor_bytes(R, rks[r])))
            L, R = R, xor_bytes(L, F)
        return L + R

    def decrypt_block(self, ct: bytes, key: bytes) -> bytes:
        if len(ct) != self.block_size_bytes:
            raise ValueError(\"bad ct length\")
        if len(key) != self.key_size_bytes:
            raise ValueError(\"bad key length\")
        if self.block_size_bytes % 2 != 0:
            raise ValueError(\"Feistel needs even byte blocks\")
        half = self.block_size_bytes // 2
        rks = ks_sha256_kdf(key, rounds=self.rounds, out_len=half, seed=self.seed)
        L, R = ct[:half], ct[half:]
        for r in reversed(range(self.rounds)):
            prev_R = L
            F = {perm_f}({sbox_f}(xor_bytes(prev_R, rks[r])))
            prev_L = xor_bytes(R, F)
            L, R = prev_L, prev_R
        return L + R

def self_test() -> None:
    import os
    c = Cipher()
    for _ in range(50):
        key = os.urandom(c.key_size_bytes)
        pt = os.urandom(c.block_size_bytes)
        ct = c.encrypt_block(pt, key)
        rt = c.decrypt_block(ct, key)
        assert rt == pt, (pt.hex(), ct.hex(), rt.hex())
    print(\"self_test OK\")
"""
    elif spec.architecture == "ARX":
        arch_code = f"""@dataclass
class Cipher:
    rounds: int = {spec.rounds}
    block_size_bytes: int = {spec.block_size_bits // 8}
    key_size_bytes: int = {spec.key_size_bits // 8}
    seed: int = {spec.seed}

    def _arx_add(self, data: bytes) -> bytes:
        words = bytes_to_words(data, 4, 'little')
        result = []
        for i in range(0, len(words), 2):
            if i + 1 < len(words):
                result.append((words[i] + words[i + 1]) & 0xFFFFFFFF)
                result.append(words[i + 1])
            else:
                result.append(words[i])
        return words_to_bytes(result, 4, 'little')

    def _arx_sub(self, data: bytes) -> bytes:
        words = bytes_to_words(data, 4, 'little')
        result = []
        for i in range(0, len(words), 2):
            if i + 1 < len(words):
                result.append((words[i] - words[i + 1]) & 0xFFFFFFFF)
                result.append(words[i + 1])
            else:
                result.append(words[i])
        return words_to_bytes(result, 4, 'little')

    def _arx_rot_fwd(self, data: bytes) -> bytes:
        words = bytes_to_words(data, 4, 'little')
        return words_to_bytes([rotate_left(w, 3, 32) for w in words], 4, 'little')

    def _arx_rot_inv(self, data: bytes) -> bytes:
        words = bytes_to_words(data, 4, 'little')
        return words_to_bytes([rotate_right(w, 3, 32) for w in words], 4, 'little')

    def encrypt_block(self, pt: bytes, key: bytes) -> bytes:
        if len(pt) != self.block_size_bytes:
            raise ValueError(\"bad pt length\")
        if len(key) != self.key_size_bytes:
            raise ValueError(\"bad key length\")
        rks = ks_sha256_kdf(key, rounds=self.rounds, out_len=self.block_size_bytes, seed=self.seed)
        state = pt
        for r in range(self.rounds):
            state = xor_bytes(state, rks[r])
            state = self._arx_add(state)
            state = self._arx_rot_fwd(state)
        state = xor_bytes(state, rks[self.rounds])
        return state

    def decrypt_block(self, ct: bytes, key: bytes) -> bytes:
        if len(ct) != self.block_size_bytes:
            raise ValueError(\"bad ct length\")
        if len(key) != self.key_size_bytes:
            raise ValueError(\"bad key length\")
        rks = ks_sha256_kdf(key, rounds=self.rounds, out_len=self.block_size_bytes, seed=self.seed)
        state = xor_bytes(ct, rks[self.rounds])
        for r in reversed(range(self.rounds)):
            state = self._arx_rot_inv(state)
            state = self._arx_sub(state)
            state = xor_bytes(state, rks[r])
        return state

def self_test() -> None:
    import os
    c = Cipher()
    for _ in range(50):
        key = os.urandom(c.key_size_bytes)
        pt = os.urandom(c.block_size_bytes)
        ct = c.encrypt_block(pt, key)
        rt = c.decrypt_block(ct, key)
        assert rt == pt, (pt.hex(), ct.hex(), rt.hex())
    print(\"self_test OK\")
"""
    else:
        raise ValueError(f"Unsupported architecture for exporter: {spec.architecture}")

    return header + util + "\n" + sbox_code + "\n" + feistel_f_code + "\n" + perm_code + "\n" + linear_code + "\n" + arch_code
