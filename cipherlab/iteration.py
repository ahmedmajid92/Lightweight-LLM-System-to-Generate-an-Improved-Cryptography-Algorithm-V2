"""Iteration history for the closed-loop improvement workflow.

Tracks each generate → preview → apply → evaluate → accept/reject cycle
with flat metric summaries for efficient comparison and display.

Research / education only. Do NOT use in production.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Flat metric summary (Option A)
# ---------------------------------------------------------------------------

class MetricsSummary(BaseModel):
    """Flat summary of key cipher evaluation metrics.

    Designed for before/after comparison in the UI and chat context.
    Full EvaluationReport data is persisted separately to disk when saving
    reproducible runs.
    """
    pt_avalanche_mean: Optional[float] = Field(default=None, description="Plaintext avalanche mean (ideal 0.5)")
    key_avalanche_mean: Optional[float] = Field(default=None, description="Key avalanche mean (ideal 0.5)")
    pt_avalanche_score: Optional[float] = Field(default=None, description="Plaintext avalanche score (0-1)")
    key_avalanche_score: Optional[float] = Field(default=None, description="Key avalanche score (0-1)")
    overall_score: Optional[float] = Field(default=None, description="Combined avalanche score (0-1)")
    sac_deviation_pt: Optional[float] = Field(default=None, description="SAC deviation for plaintext (ideal 0.0)")
    sac_deviation_key: Optional[float] = Field(default=None, description="SAC deviation for key (ideal 0.0)")
    sac_passes_pt: Optional[bool] = Field(default=None, description="Whether plaintext SAC passes heuristic")
    sac_passes_key: Optional[bool] = Field(default=None, description="Whether key SAC passes heuristic")
    roundtrip_pass: Optional[bool] = Field(default=None, description="Whether roundtrip P=D(E(P,K),K) holds")
    roundtrip_fail_count: Optional[int] = Field(default=None, description="Number of failed roundtrip vectors")
    num_heuristic_issues: int = Field(default=0, description="Count of heuristic issues detected")

    def delta(self, other: MetricsSummary) -> Dict[str, Optional[float]]:
        """Compute metric deltas: self (after) - other (before).

        Positive delta = improvement for scores (higher is better).
        Negative delta = improvement for deviations (lower is better).
        Returns None for fields where either side is None.
        """
        deltas: Dict[str, Optional[float]] = {}
        for field_name in (
            "pt_avalanche_mean", "key_avalanche_mean",
            "pt_avalanche_score", "key_avalanche_score", "overall_score",
            "sac_deviation_pt", "sac_deviation_key",
        ):
            after_val = getattr(self, field_name)
            before_val = getattr(other, field_name)
            if after_val is not None and before_val is not None:
                deltas[field_name] = round(after_val - before_val, 6)
            else:
                deltas[field_name] = None
        return deltas

    def to_display_dict(self) -> Dict[str, Any]:
        """Return a compact dict suitable for UI display."""
        return {k: v for k, v in self.model_dump().items() if v is not None}


def extract_metrics_summary(
    metrics: Optional[Dict[str, Any]] = None,
    eval_report: Optional[Any] = None,
    issues: Optional[List[str]] = None,
) -> MetricsSummary:
    """Build a MetricsSummary from existing evaluation outputs.

    Args:
        metrics: Dict returned by evaluate_and_score().
        eval_report: An EvaluationReport instance (optional, for SAC/roundtrip).
        issues: List of heuristic issue strings.

    Returns:
        Populated MetricsSummary.
    """
    summary = MetricsSummary()

    if metrics:
        pt_av = metrics.get("plaintext_avalanche", {})
        key_av = metrics.get("key_avalanche", {})
        scores = metrics.get("scores", {})

        if isinstance(pt_av, dict):
            summary.pt_avalanche_mean = pt_av.get("mean")
        if isinstance(key_av, dict):
            summary.key_avalanche_mean = key_av.get("mean")
        if isinstance(scores, dict):
            summary.pt_avalanche_score = scores.get("plaintext_avalanche")
            summary.key_avalanche_score = scores.get("key_avalanche")
            summary.overall_score = scores.get("overall")

    if eval_report is not None:
        # Roundtrip
        if hasattr(eval_report, "roundtrip_results"):
            for rt in eval_report.roundtrip_results:
                summary.roundtrip_pass = rt.is_perfect
                summary.roundtrip_fail_count = rt.failed
        # SAC
        if hasattr(eval_report, "sac_results"):
            for sac in eval_report.sac_results:
                if sac.input_type == "plaintext":
                    summary.sac_deviation_pt = sac.sac_deviation
                    summary.sac_passes_pt = sac.passes_sac
                elif sac.input_type == "key":
                    summary.sac_deviation_key = sac.sac_deviation
                    summary.sac_passes_key = sac.passes_sac

    if issues is not None:
        summary.num_heuristic_issues = len(issues)

    return summary


# ---------------------------------------------------------------------------
# Iteration record
# ---------------------------------------------------------------------------

class IterationRecord(BaseModel):
    """One cycle of the closed-loop improvement workflow.

    Captures the full lifecycle: generate patch → apply → evaluate → decide.
    """
    iteration_id: int = Field(..., ge=0, description="Zero-based iteration index")
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    # Spec snapshots (serialized)
    before_spec: Dict[str, Any] = Field(default_factory=dict, description="CipherSpec dict before patch")
    after_spec: Optional[Dict[str, Any]] = Field(default=None, description="CipherSpec dict after patch applied")

    # Patch
    patch: Optional[Dict[str, Any]] = Field(default=None, description="ImprovementPatch dict")
    patch_summary: str = Field(default="", description="One-line patch description")

    # Evaluation
    before_metrics: Optional[MetricsSummary] = None
    after_metrics: Optional[MetricsSummary] = None
    metric_deltas: Optional[Dict[str, Optional[float]]] = None
    validation_ok: Optional[bool] = Field(default=None, description="Whether patched spec passed validation")
    validation_errors: List[str] = Field(default_factory=list)

    # Model info
    model_used: str = Field(default="", description="Model ID that generated the patch")
    reasoning_trace: Optional[str] = Field(default=None, description="DeepSeek-R1 reasoning trace (truncated)")

    # Decision
    status: Literal["pending", "accepted", "rejected"] = Field(default="pending")
    decision_reason: str = Field(default="", description="Human-provided reason for accept/reject")

    # Reproducibility
    seed: Optional[int] = None

    def compute_deltas(self) -> None:
        """Compute and store metric_deltas from before/after metrics."""
        if self.after_metrics and self.before_metrics:
            self.metric_deltas = self.after_metrics.delta(self.before_metrics)

    def summary_line(self) -> str:
        """One-line summary for display in history tables."""
        status_icon = {"pending": "?", "accepted": "+", "rejected": "-"}.get(self.status, "?")
        score_info = ""
        if self.metric_deltas and self.metric_deltas.get("overall_score") is not None:
            d = self.metric_deltas["overall_score"]
            sign = "+" if d >= 0 else ""
            score_info = f" overall={sign}{d:.4f}"
        return f"[{status_icon}] #{self.iteration_id}: {self.patch_summary}{score_info} ({self.model_used})"


# ---------------------------------------------------------------------------
# Iteration history
# ---------------------------------------------------------------------------

class IterationHistory(BaseModel):
    """Ordered history of improvement iterations for one cipher design session."""
    cipher_name: str = ""
    records: List[IterationRecord] = Field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.records)

    @property
    def next_id(self) -> int:
        return len(self.records)

    def current_spec_dict(self) -> Optional[Dict[str, Any]]:
        """Return the spec dict from the last accepted iteration, or None."""
        for rec in reversed(self.records):
            if rec.status == "accepted" and rec.after_spec:
                return rec.after_spec
        return None

    def add(self, record: IterationRecord) -> None:
        self.records.append(record)

    def accepted(self) -> List[IterationRecord]:
        return [r for r in self.records if r.status == "accepted"]

    def rejected(self) -> List[IterationRecord]:
        return [r for r in self.records if r.status == "rejected"]

    def pending(self) -> List[IterationRecord]:
        return [r for r in self.records if r.status == "pending"]

    def get(self, iteration_id: int) -> Optional[IterationRecord]:
        for r in self.records:
            if r.iteration_id == iteration_id:
                return r
        return None

    def rollback_spec_dict(self, to_iteration_id: int) -> Optional[Dict[str, Any]]:
        """Get the after_spec from a specific accepted iteration for rollback.

        If to_iteration_id is -1, returns None (meaning: revert to original spec).
        """
        if to_iteration_id < 0:
            return None
        rec = self.get(to_iteration_id)
        if rec and rec.status == "accepted" and rec.after_spec:
            return rec.after_spec
        return None

    def to_context_summary(self, max_records: int = 10) -> str:
        """Format history as concise text for LLM prompt injection.

        Used by the design-review copilot to answer questions about
        the iteration history.
        """
        if not self.records:
            return "No improvement iterations yet."

        lines = [f"Iteration history ({len(self.records)} total):"]
        recent = self.records[-max_records:]
        for rec in recent:
            lines.append(rec.summary_line())
            if rec.decision_reason:
                lines.append(f"  Reason: {rec.decision_reason}")
            if rec.metric_deltas:
                notable = {
                    k: v for k, v in rec.metric_deltas.items()
                    if v is not None and abs(v) > 0.001
                }
                if notable:
                    delta_strs = [f"{k}={v:+.4f}" for k, v in notable.items()]
                    lines.append(f"  Deltas: {', '.join(delta_strs)}")
        return "\n".join(lines)

    def to_export_dict(self) -> Dict[str, Any]:
        """Full serializable dict for disk persistence."""
        return {
            "cipher_name": self.cipher_name,
            "total_iterations": self.count,
            "accepted_count": len(self.accepted()),
            "rejected_count": len(self.rejected()),
            "records": [r.model_dump() for r in self.records],
        }
