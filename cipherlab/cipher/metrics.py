from __future__ import annotations

from typing import Dict, List, Tuple

from .builder import build_cipher
from .cryptanalysis import evaluate_cipher
from .spec import CipherSpec


def score_avalanche(mean: float) -> float:
    # 1.0 is perfect (0.5), 0.0 is terrible (0 or 1)
    return max(0.0, 1.0 - abs(mean - 0.5) / 0.5)


def evaluate_and_score(spec: CipherSpec) -> Dict[str, object]:
    cipher = build_cipher(spec)
    metrics = evaluate_cipher(
        cipher,
        block_size_bits=spec.block_size_bits,
        key_size_bits=spec.key_size_bits,
        rounds=spec.rounds,
        seed=spec.seed,
    )
    pt_mean = float(metrics["plaintext_avalanche"]["mean"])
    key_mean = float(metrics["key_avalanche"]["mean"])

    metrics["scores"] = {
        "plaintext_avalanche": score_avalanche(pt_mean),
        "key_avalanche": score_avalanche(key_mean),
        "overall": (score_avalanche(pt_mean) + score_avalanche(key_mean)) / 2.0,
    }
    return metrics


def heuristic_issues(metrics: Dict[str, object]) -> List[str]:
    issues: List[str] = []
    pt = float(metrics["plaintext_avalanche"]["mean"])
    kk = float(metrics["key_avalanche"]["mean"])
    if pt < 0.40:
        issues.append("Plaintext avalanche is low (<0.40). Diffusion likely insufficient.")
    if kk < 0.40:
        issues.append("Key avalanche is low (<0.40). Key schedule or mixing may be weak.")
    return issues
