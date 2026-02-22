"""Cryptographic dataset compilation for open-source release.

Exports benchmark experiment results as JSONL (JSON Lines) files containing
CipherSpecs, evolved component ASTs, DeepSeek-R1 reasoning traces, and
evaluation scores.

Research / education only. Do NOT use in production.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from .benchmark_runner import ExperimentResult


# ---------------------------------------------------------------------------
# Record builder
# ---------------------------------------------------------------------------

def _build_record(exp: ExperimentResult) -> Dict[str, Any]:
    """Convert an ExperimentResult to a compact dataset record dict.

    Extracts the essential fields needed for an open-source release
    without the full per-iteration history.
    """
    # Collect reasoning traces from all iterations
    reasoning_traces: List[str] = []
    for m in exp.iteration_history:
        if m.reasoning_trace:
            reasoning_traces.append(m.reasoning_trace)

    # Get S-box metrics from last iteration
    sbox_ddt_max: Optional[int] = None
    sbox_lat_max: Optional[int] = None
    if exp.iteration_history:
        last = exp.iteration_history[-1]
        sbox_ddt_max = last.sbox_ddt_max
        sbox_lat_max = last.sbox_lat_max

    # Check if roundtrip passes in final iteration
    roundtrip_pass = False
    if exp.iteration_history:
        roundtrip_pass = exp.iteration_history[-1].roundtrip_passed

    return {
        # Identity
        "experiment_id": exp.experiment_id,
        "algorithm": exp.algorithm,
        "architecture": exp.architecture,
        "model_label": exp.model_label,
        "model_id": exp.model_id,
        "seed": exp.seed,
        # Final cipher state
        "final_spec": exp.final_spec or {},
        "final_components": exp.final_components or {},
        # Evaluation scores
        "roundtrip_pass": roundtrip_pass,
        "sac_pt_mean": exp.final_sac_pt_mean,
        "sac_pt_deviation": exp.final_sac_pt_deviation,
        "sac_key_mean": exp.final_sac_key_mean,
        "sac_key_deviation": exp.final_sac_key_deviation,
        "sbox_ddt_max": sbox_ddt_max,
        "sbox_lat_max": sbox_lat_max,
        # Reasoning traces
        "reasoning_traces": reasoning_traces,
        # Summary metrics
        "iteration_count": exp.iterations_run,
        "iterations_to_roundtrip_pass": exp.iterations_to_roundtrip_pass,
        "total_evolutions": exp.total_evolutions_succeeded,
        "total_tokens": exp.total_tokens_input + exp.total_tokens_output,
        "total_time_seconds": exp.total_time_seconds,
    }


# ---------------------------------------------------------------------------
# Export functions
# ---------------------------------------------------------------------------

def export_dataset_jsonl(
    experiments: List[ExperimentResult],
    output_path: Path | str,
) -> Path:
    """Write compact JSONL dataset (one record per experiment).

    Each line is a self-contained JSON object with final cipher state,
    evaluation scores, and reasoning traces. Suitable for open-source release.

    Args:
        experiments: List of completed experiment results.
        output_path: Path for the output .jsonl file.

    Returns:
        Path to the created file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for exp in experiments:
            if exp.error and not exp.iteration_history:
                continue  # Skip completely failed experiments
            record = _build_record(exp)
            f.write(json.dumps(record, sort_keys=True) + "\n")

    return output_path


def export_full_jsonl(
    experiments: List[ExperimentResult],
    output_path: Path | str,
) -> Path:
    """Write detailed JSONL including per-iteration history.

    Each line includes the full iteration_history with per-step metrics,
    patch details, and token usage. Larger file but complete reproducibility.

    Args:
        experiments: List of completed experiment results.
        output_path: Path for the output .jsonl file.

    Returns:
        Path to the created file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for exp in experiments:
            record = exp.to_dict()
            f.write(json.dumps(record, sort_keys=True) + "\n")

    return output_path
