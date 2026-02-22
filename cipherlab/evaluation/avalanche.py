"""Strict Avalanche Criterion (SAC) calculator with per-bit analysis.

Measures whether flipping each individual input bit causes each output bit
to flip with probability ~0.5. A cipher satisfying SAC has good diffusion.

Research / education only. Do NOT use in production.
"""
from __future__ import annotations

import random
import statistics
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional

from cipherlab.cipher.cryptanalysis import (
    _hamming_distance_bytes,
    _flip_bit,
    _rand_bytes,
)
from cipherlab.cipher.builder import BlockCipher


@dataclass
class SACResult:
    """Strict Avalanche Criterion measurement for one input type."""
    algorithm_name: str
    architecture: str
    input_type: str             # "plaintext" or "key"
    num_trials: int
    num_input_bits: int
    num_output_bits: int

    # Per-input-bit mean flip fraction (len = num_input_bits)
    per_input_bit_mean: List[float] = field(default_factory=list)

    # Overall statistics
    global_mean: float = 0.0    # Mean across all per-bit means (~0.5 ideal)
    global_std: float = 0.0     # Std dev of per-bit means (lower = more uniform)
    min_bit_prob: float = 0.0   # Lowest per-bit mean
    max_bit_prob: float = 0.0   # Highest per-bit mean
    sac_deviation: float = 0.0  # Mean |per_bit - 0.5| (0.0 = perfect SAC)

    @property
    def passes_sac(self) -> bool:
        """Heuristic: SAC deviation < 0.05 and min_bit_prob > 0.35."""
        return self.sac_deviation < 0.05 and self.min_bit_prob > 0.35

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["passes_sac"] = self.passes_sac
        return d

    def summary(self) -> str:
        status = "PASS" if self.passes_sac else "FAIL"
        return (
            f"[{status}] SAC({self.input_type}): "
            f"mean={self.global_mean:.4f}, std={self.global_std:.4f}, "
            f"deviation={self.sac_deviation:.4f}, "
            f"min={self.min_bit_prob:.4f}, max={self.max_bit_prob:.4f}"
        )


def compute_sac(
    cipher: BlockCipher,
    *,
    block_size_bits: int,
    key_size_bits: int,
    input_type: str = "plaintext",
    trials: int = 500,
    seed: int = 1337,
    algorithm_name: str = "",
    architecture: str = "",
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> SACResult:
    """Compute Strict Avalanche Criterion with per-input-bit analysis.

    For each input bit position i:
      - Run `trials` iterations with random inputs
      - Flip bit i, encrypt both, measure output Hamming distance
      - Record mean fraction of output bits that flipped

    Args:
        cipher: Built cipher with encrypt_block method.
        block_size_bits: Block size in bits.
        key_size_bits: Key size in bits.
        input_type: "plaintext" or "key" — which input to perturb.
        trials: Number of random trials per input bit.
        seed: Random seed for reproducibility.
        algorithm_name: Name for labeling results.
        architecture: Architecture type for labeling results.
        progress_callback: Optional callback(current_bit, total_bits).

    Returns:
        SACResult with per-bit and aggregate statistics.
    """
    block_bytes = block_size_bits // 8
    key_bytes = key_size_bits // 8

    if input_type == "plaintext":
        num_input_bits = block_size_bits
    elif input_type == "key":
        num_input_bits = key_size_bits
    else:
        raise ValueError(f"input_type must be 'plaintext' or 'key', got '{input_type}'")

    num_output_bits = block_size_bits
    rng = random.Random(seed)

    per_bit_means: List[float] = []

    for bit_i in range(num_input_bits):
        if progress_callback:
            progress_callback(bit_i, num_input_bits)

        total_frac = 0.0
        for _ in range(trials):
            pt = _rand_bytes(rng, block_bytes)
            key = _rand_bytes(rng, key_bytes)

            ct1 = cipher.encrypt_block(pt, key)

            if input_type == "plaintext":
                pt2 = _flip_bit(pt, bit_i)
                ct2 = cipher.encrypt_block(pt2, key)
            else:
                key2 = _flip_bit(key, bit_i)
                ct2 = cipher.encrypt_block(pt, key2)

            dist = _hamming_distance_bytes(ct1, ct2)
            total_frac += dist / num_output_bits

        per_bit_means.append(total_frac / trials)

    # Compute aggregate statistics
    global_mean = statistics.mean(per_bit_means) if per_bit_means else 0.0
    global_std = statistics.stdev(per_bit_means) if len(per_bit_means) > 1 else 0.0
    min_bit = min(per_bit_means) if per_bit_means else 0.0
    max_bit = max(per_bit_means) if per_bit_means else 0.0
    sac_dev = statistics.mean(abs(p - 0.5) for p in per_bit_means) if per_bit_means else 0.5

    return SACResult(
        algorithm_name=algorithm_name,
        architecture=architecture,
        input_type=input_type,
        num_trials=trials,
        num_input_bits=num_input_bits,
        num_output_bits=num_output_bits,
        per_input_bit_mean=per_bit_means,
        global_mean=round(global_mean, 6),
        global_std=round(global_std, 6),
        min_bit_prob=round(min_bit, 6),
        max_bit_prob=round(max_bit, 6),
        sac_deviation=round(sac_dev, 6),
    )
