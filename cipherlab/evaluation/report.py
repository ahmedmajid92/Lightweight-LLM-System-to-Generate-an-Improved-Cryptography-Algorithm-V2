"""Structured evaluation report builder.

Aggregates results from roundtrip tests, SAC analysis, and S-box analysis
into a single serializable report for export and UI display.

Research / education only. Do NOT use in production.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from .roundtrip import RoundtripResult
from .avalanche import SACResult
from .sbox_analysis import SBoxAnalysisResult


@dataclass
class EvaluationReport:
    """Complete evaluation report aggregating all analysis results."""
    timestamp: str = ""
    roundtrip_results: List[RoundtripResult] = field(default_factory=list)
    sac_results: List[SACResult] = field(default_factory=list)
    sbox_results: List[SBoxAnalysisResult] = field(default_factory=list)
    feedback: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.utcnow().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize full report for JSON export."""
        return {
            "timestamp": self.timestamp,
            "roundtrip": [r.to_dict() for r in self.roundtrip_results],
            "sac": [s.to_dict() for s in self.sac_results],
            "sbox": [s.to_dict() for s in self.sbox_results],
            "feedback": self.feedback,
            "summary": {
                "total_algorithms_tested": len(self.roundtrip_results),
                "roundtrip_all_pass": all(r.is_perfect for r in self.roundtrip_results),
                "sac_all_pass": all(s.passes_sac for s in self.sac_results),
                "failing_algorithms": self.failing_algorithms(),
            },
        }

    def to_summary(self) -> str:
        """Human-readable summary for Streamlit display."""
        lines = [f"Evaluation Report — {self.timestamp}", "=" * 50]

        # Roundtrip summary
        if self.roundtrip_results:
            rt_pass = sum(1 for r in self.roundtrip_results if r.is_perfect)
            rt_total = len(self.roundtrip_results)
            lines.append(f"\nRoundtrip Tests: {rt_pass}/{rt_total} algorithms pass")
            for r in self.roundtrip_results:
                lines.append(f"  {r.summary()}")

        # SAC summary
        if self.sac_results:
            sac_pass = sum(1 for s in self.sac_results if s.passes_sac)
            sac_total = len(self.sac_results)
            lines.append(f"\nSAC Analysis: {sac_pass}/{sac_total} pass")
            for s in self.sac_results:
                lines.append(f"  {s.summary()}")

        # S-box summary
        if self.sbox_results:
            lines.append(f"\nS-box Analysis: {len(self.sbox_results)} components")
            for s in self.sbox_results:
                lines.append(f"  {s.summary()}")

        return "\n".join(lines)

    def failing_algorithms(self) -> List[str]:
        """Return names of algorithms with roundtrip failures."""
        return [r.algorithm_name for r in self.roundtrip_results if not r.is_perfect]

    def weak_sac_algorithms(self) -> List[str]:
        """Return names of algorithms that fail SAC heuristic."""
        return [s.algorithm_name for s in self.sac_results if not s.passes_sac]
