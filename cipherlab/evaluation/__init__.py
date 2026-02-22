"""Deterministic Cryptographic Evaluation Framework.

Provides algebraic unit testing (roundtrip verification), statistical
analysis (SAC, DDT/LAT), and autonomous feedback synthesis via LLM.

Research / education only. Do NOT use in production.
"""

from .roundtrip import RoundtripResult, RoundtripFailure, run_roundtrip_tests, run_all_algorithms
from .avalanche import SACResult, compute_sac
from .sbox_analysis import SBoxAnalysisResult, analyze_sbox, analyze_all_sboxes
from .report import EvaluationReport
from .feedback import (
    EvaluationDiagnostic,
    FeedbackCycleResult,
    parse_evaluation_results,
    build_feedback_prompt,
    run_feedback_cycle,
)

__all__ = [
    "RoundtripResult",
    "RoundtripFailure",
    "run_roundtrip_tests",
    "run_all_algorithms",
    "SACResult",
    "compute_sac",
    "SBoxAnalysisResult",
    "analyze_sbox",
    "analyze_all_sboxes",
    "EvaluationReport",
    "EvaluationDiagnostic",
    "FeedbackCycleResult",
    "parse_evaluation_results",
    "build_feedback_prompt",
    "run_feedback_cycle",
]
