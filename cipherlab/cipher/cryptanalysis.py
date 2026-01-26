from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .builder import BlockCipher


def _hamming_distance_bytes(a: bytes, b: bytes) -> int:
    if len(a) != len(b):
        raise ValueError("hamming distance length mismatch")
    dist = 0
    for x, y in zip(a, b):
        dist += (x ^ y).bit_count()
    return dist


def _flip_bit(data: bytes, bit_index: int) -> bytes:
    byte_i = bit_index // 8
    bit_i = bit_index % 8
    if byte_i < 0 or byte_i >= len(data):
        raise IndexError("bit_index out of range")
    mask = 1 << bit_i
    out = bytearray(data)
    out[byte_i] ^= mask
    return bytes(out)


def _rand_bytes(rng: random.Random, n: int) -> bytes:
    return bytes(rng.randrange(0, 256) for _ in range(n))


def avalanche_plaintext(
    cipher: BlockCipher,
    *,
    block_size_bytes: int,
    key_size_bytes: int,
    trials: int = 200,
    flips_per_trial: int = 1,
    seed: int = 1337,
) -> Dict[str, float]:
    rng = random.Random(seed)
    total_frac = 0.0
    total_bits = block_size_bytes * 8
    for _ in range(trials):
        key = _rand_bytes(rng, key_size_bytes)
        pt = _rand_bytes(rng, block_size_bytes)
        ct = cipher.encrypt_block(pt, key)
        for _ in range(flips_per_trial):
            bit = rng.randrange(0, total_bits)
            pt2 = _flip_bit(pt, bit)
            ct2 = cipher.encrypt_block(pt2, key)
            total_frac += _hamming_distance_bytes(ct, ct2) / total_bits
    denom = trials * flips_per_trial
    return {
        "mean": total_frac / denom if denom else 0.0,
    }


def avalanche_key(
    cipher: BlockCipher,
    *,
    block_size_bytes: int,
    key_size_bytes: int,
    trials: int = 200,
    flips_per_trial: int = 1,
    seed: int = 1337,
) -> Dict[str, float]:
    rng = random.Random(seed + 1)
    total_frac = 0.0
    total_bits = block_size_bytes * 8
    key_bits = key_size_bytes * 8
    for _ in range(trials):
        key = _rand_bytes(rng, key_size_bytes)
        pt = _rand_bytes(rng, block_size_bytes)
        ct = cipher.encrypt_block(pt, key)
        for _ in range(flips_per_trial):
            bit = rng.randrange(0, key_bits)
            key2 = _flip_bit(key, bit)
            ct2 = cipher.encrypt_block(pt, key2)
            total_frac += _hamming_distance_bytes(ct, ct2) / total_bits
    denom = trials * flips_per_trial
    return {
        "mean": total_frac / denom if denom else 0.0,
    }


def sbox_ddt_max(sbox: List[int]) -> int:
    """Return max entry in DDT excluding dx=0 (scaled by counts, not prob)."""
    n = len(sbox)
    if n not in (16, 256):
        raise ValueError("sbox must be 4-bit (16) or 8-bit (256)")
    max_v = 0
    for dx in range(1, n):
        # distribution of dy
        counts = {}
        for x in range(n):
            dy = sbox[x] ^ sbox[x ^ dx]
            counts[dy] = counts.get(dy, 0) + 1
        local_max = max(counts.values())
        if local_max > max_v:
            max_v = local_max
    return max_v


def sbox_lat_max_abs(sbox: List[int]) -> int:
    """Return max absolute bias*2^m (Walsh) for non-trivial masks."""
    n = len(sbox)
    m = int(math.log2(n))
    if 2**m != n:
        raise ValueError("sbox size must be power of 2")
    max_abs = 0
    for a in range(1, n):
        for b in range(1, n):
            s = 0
            for x in range(n):
                ax = (a & x).bit_count() % 2
                bx = (b & sbox[x]).bit_count() % 2
                s += 1 if ax == bx else -1
            max_abs = max(max_abs, abs(s))
    return max_abs


def evaluate_cipher(cipher: BlockCipher, *, block_size_bits: int, key_size_bits: int, rounds: int, seed: int) -> Dict[str, object]:
    bs = block_size_bits // 8
    ks = key_size_bits // 8
    pt = avalanche_plaintext(cipher, block_size_bytes=bs, key_size_bytes=ks, trials=200, flips_per_trial=1, seed=seed)
    kk = avalanche_key(cipher, block_size_bytes=bs, key_size_bytes=ks, trials=200, flips_per_trial=1, seed=seed)
    return {
        "block_size_bits": block_size_bits,
        "key_size_bits": key_size_bits,
        "rounds": rounds,
        "plaintext_avalanche": pt,
        "key_avalanche": kk,
    }
