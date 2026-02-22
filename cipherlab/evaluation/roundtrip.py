"""Algebraic unit testing: roundtrip verification P = D(E(P, K), K).

Generates thousands of randomized test vectors per algorithm and verifies
that decryption perfectly inverts encryption for every vector.

Research / education only. Do NOT use in production.
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional

import sys
from pathlib import Path

# Add project root so we can import AlgorithmsBlock
_project_root = Path(__file__).parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from AlgorithmsBlock import (
    CipherSpec,
    build_cipher,
    get_template,
    list_algorithms,
    ComponentRegistry,
)


@dataclass
class RoundtripFailure:
    """Details of a single failed roundtrip test vector."""
    vector_index: int
    plaintext_hex: str
    key_hex: str
    ciphertext_hex: str
    decrypted_hex: str       # What decrypt returned (should equal plaintext)
    error: Optional[str]     # Exception message if decrypt/encrypt threw


@dataclass
class RoundtripResult:
    """Aggregate result of roundtrip testing for one algorithm."""
    algorithm_name: str
    architecture: str
    block_size_bits: int
    key_size_bits: int
    rounds: int
    total_vectors: int
    passed: int
    failed: int
    failures: List[RoundtripFailure] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    seed: int = 1337

    @property
    def success_rate(self) -> float:
        return self.passed / self.total_vectors if self.total_vectors > 0 else 0.0

    @property
    def is_perfect(self) -> bool:
        return self.failed == 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def summary(self) -> str:
        status = "PASS" if self.is_perfect else "FAIL"
        return (
            f"[{status}] {self.algorithm_name} ({self.architecture}): "
            f"{self.passed}/{self.total_vectors} vectors passed "
            f"({self.elapsed_seconds:.2f}s)"
        )


def _rand_bytes(rng: random.Random, n: int) -> bytes:
    return bytes(rng.randrange(0, 256) for _ in range(n))


def run_roundtrip_tests(
    spec: CipherSpec,
    *,
    num_vectors: int = 1000,
    seed: int = 1337,
    max_failures_recorded: int = 10,
    registry: Optional[ComponentRegistry] = None,
) -> RoundtripResult:
    """Run roundtrip verification P = D(E(P, K), K) across many test vectors.

    Args:
        spec: Cipher specification to test.
        num_vectors: Number of random (plaintext, key) pairs to test.
        seed: Random seed for deterministic reproducibility.
        max_failures_recorded: Maximum number of failure details to keep.
        registry: Optional component registry; uses default if not provided.

    Returns:
        RoundtripResult with pass/fail counts and failure details.
    """
    reg = registry or ComponentRegistry()
    cipher = build_cipher(spec, reg)

    block_bytes = spec.block_size_bits // 8
    key_bytes = spec.key_size_bits // 8

    rng = random.Random(seed)
    passed = 0
    failed = 0
    failures: List[RoundtripFailure] = []

    start = time.perf_counter()

    for i in range(num_vectors):
        pt = _rand_bytes(rng, block_bytes)
        key = _rand_bytes(rng, key_bytes)

        try:
            ct = cipher.encrypt_block(pt, key)
            pt2 = cipher.decrypt_block(ct, key)

            if pt == pt2:
                passed += 1
            else:
                failed += 1
                if len(failures) < max_failures_recorded:
                    failures.append(RoundtripFailure(
                        vector_index=i,
                        plaintext_hex=pt.hex(),
                        key_hex=key.hex(),
                        ciphertext_hex=ct.hex(),
                        decrypted_hex=pt2.hex(),
                        error=None,
                    ))
        except Exception as exc:
            failed += 1
            if len(failures) < max_failures_recorded:
                failures.append(RoundtripFailure(
                    vector_index=i,
                    plaintext_hex=pt.hex(),
                    key_hex=key.hex(),
                    ciphertext_hex="<error>",
                    decrypted_hex="<error>",
                    error=str(exc),
                ))

    elapsed = time.perf_counter() - start

    return RoundtripResult(
        algorithm_name=spec.name,
        architecture=spec.architecture,
        block_size_bits=spec.block_size_bits,
        key_size_bits=spec.key_size_bits,
        rounds=spec.rounds,
        total_vectors=num_vectors,
        passed=passed,
        failed=failed,
        failures=failures,
        elapsed_seconds=round(elapsed, 4),
        seed=seed,
    )


def run_all_algorithms(
    *,
    num_vectors: int = 1000,
    seed: int = 1337,
    progress_callback: Optional[Callable[[str, int, int], None]] = None,
) -> List[RoundtripResult]:
    """Run roundtrip tests for all 12 algorithms in ALGORITHM_LIBRARY.

    Args:
        num_vectors: Number of test vectors per algorithm.
        seed: Random seed for reproducibility.
        progress_callback: Optional callback(algo_name, current_index, total)
            for progress reporting (e.g., Streamlit).

    Returns:
        List of RoundtripResult sorted by algorithm name.
    """
    algos = list_algorithms()
    registry = ComponentRegistry()
    results: List[RoundtripResult] = []

    for idx, algo_name in enumerate(algos):
        if progress_callback:
            progress_callback(algo_name, idx, len(algos))

        spec = get_template(algo_name, seed=seed)
        result = run_roundtrip_tests(
            spec,
            num_vectors=num_vectors,
            seed=seed,
            registry=registry,
        )
        results.append(result)

    return sorted(results, key=lambda r: r.algorithm_name)
