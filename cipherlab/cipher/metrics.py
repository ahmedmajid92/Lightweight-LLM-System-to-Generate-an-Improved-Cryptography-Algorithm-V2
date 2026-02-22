from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

from .builder import build_cipher
from .cryptanalysis import evaluate_cipher
from .registry import ComponentRegistry
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


def evaluate_full(
    spec: CipherSpec,
    *,
    registry: Optional[ComponentRegistry] = None,
    sac_trials: int = 500,
    seed: Optional[int] = None,
    progress_callback: Optional[Callable[[str, int, int], None]] = None,
) -> Dict[str, Any]:
    """Run comprehensive evaluation: basic avalanche + SAC + S-box analysis.

    Args:
        spec: Cipher specification to evaluate.
        registry: Optional component registry.
        sac_trials: Number of trials per bit for SAC analysis.
        seed: Random seed (defaults to spec.seed).
        progress_callback: Optional callback(stage, current, total).

    Returns:
        Dict with basic_avalanche, sac_plaintext, sac_key, sbox_analysis, and scores.
    """
    from cipherlab.evaluation.avalanche import compute_sac
    from cipherlab.evaluation.sbox_analysis import analyze_all_sboxes

    reg = registry or ComponentRegistry()
    actual_seed = seed if seed is not None else spec.seed

    # 1. Basic avalanche (existing)
    basic = evaluate_and_score(spec)

    # 2. SAC analysis
    cipher = build_cipher(spec, reg)

    if progress_callback:
        progress_callback("SAC (plaintext)", 0, 2)

    sac_pt = compute_sac(
        cipher,
        block_size_bits=spec.block_size_bits,
        key_size_bits=spec.key_size_bits,
        input_type="plaintext",
        trials=sac_trials,
        seed=actual_seed,
        algorithm_name=spec.name,
        architecture=spec.architecture,
    )

    if progress_callback:
        progress_callback("SAC (key)", 1, 2)

    sac_key = compute_sac(
        cipher,
        block_size_bits=spec.block_size_bits,
        key_size_bits=spec.key_size_bits,
        input_type="key",
        trials=sac_trials,
        seed=actual_seed,
        algorithm_name=spec.name,
        architecture=spec.architecture,
    )

    # 3. S-box analysis
    sbox_results = analyze_all_sboxes(reg)

    # 4. Composite scores
    pt_score = score_avalanche(float(basic["plaintext_avalanche"]["mean"]))
    key_score = score_avalanche(float(basic["key_avalanche"]["mean"]))

    scores = {
        "plaintext_avalanche": pt_score,
        "key_avalanche": key_score,
        "sac_deviation_pt": sac_pt.sac_deviation,
        "sac_deviation_key": sac_key.sac_deviation,
        "overall": (pt_score + key_score) / 2.0,
    }

    return {
        "basic_avalanche": basic,
        "sac_plaintext": sac_pt.to_dict(),
        "sac_key": sac_key.to_dict(),
        "sbox_analysis": [s.to_dict() for s in sbox_results],
        "scores": scores,
    }
