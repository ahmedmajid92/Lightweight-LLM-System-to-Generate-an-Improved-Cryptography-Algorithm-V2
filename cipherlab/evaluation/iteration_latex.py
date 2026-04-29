"""LaTeX table generation for iteration history thesis artifacts.

Generates publication-ready LaTeX tables summarizing the closed-loop
improvement workflow: per-iteration metrics, accept/reject decisions,
and overall improvement trajectory.

Research / education only. Do NOT use in production.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from cipherlab.iteration import IterationHistory, IterationRecord


# ---------------------------------------------------------------------------
# Helpers (compatible with latex_exporter.py patterns)
# ---------------------------------------------------------------------------

def _escape_latex(text: str) -> str:
    for ch in ["_", "&", "%", "#", "$", "{", "}"]:
        text = text.replace(ch, f"\\{ch}")
    return text


def _fmt(val: Optional[float], fmt: str = ".4f") -> str:
    if val is None:
        return "---"
    return f"${val:{fmt}}$"


def _delta_fmt(val: Optional[float], fmt: str = ".4f") -> str:
    if val is None:
        return "---"
    sign = "+" if val >= 0 else ""
    return f"${sign}{val:{fmt}}$"


def _format_table(
    headers: List[str],
    rows: List[List[str]],
    caption: str,
    label: str,
    column_format: Optional[str] = None,
) -> str:
    ncols = len(headers)
    if column_format is None:
        column_format = "l " + " ".join(["c"] * (ncols - 1))

    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{tab:{label}}}",
        f"\\begin{{tabular}}{{{column_format}}}",
        "\\toprule",
        " & ".join(headers) + " \\\\",
        "\\midrule",
    ]

    for row in rows:
        lines.append(" & ".join(row) + " \\\\")

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Table 1: Iteration summary (one row per iteration)
# ---------------------------------------------------------------------------

def iteration_summary_table(history: IterationHistory) -> str:
    """Generate a LaTeX table with one row per iteration.

    Columns: #, Status, Model, Rounds, Overall Score (before/after/delta), Decision Reason.
    """
    if not history.records:
        return "% No iterations to report."

    headers = [
        "\\#",
        "Status",
        "Model",
        "Rounds",
        "Score Before",
        "Score After",
        "$\\Delta$ Score",
        "Decision",
    ]

    rows = []
    for rec in history.records:
        status = rec.status.capitalize()
        model_short = rec.model_used.split("/")[-1] if "/" in rec.model_used else rec.model_used
        model_short = _escape_latex(model_short[:20])

        # Rounds change
        before_rounds = rec.before_spec.get("rounds", "?") if rec.before_spec else "?"
        after_rounds = rec.after_spec.get("rounds", "?") if rec.after_spec else "?"
        if before_rounds != after_rounds:
            rounds_str = f"{before_rounds}$\\to${after_rounds}"
        else:
            rounds_str = str(before_rounds)

        # Scores
        before_score = rec.before_metrics.overall_score if rec.before_metrics else None
        after_score = rec.after_metrics.overall_score if rec.after_metrics else None
        delta = (rec.metric_deltas or {}).get("overall_score")

        reason = _escape_latex(rec.decision_reason[:30]) if rec.decision_reason else "---"

        rows.append([
            str(rec.iteration_id),
            status,
            model_short,
            rounds_str,
            _fmt(before_score),
            _fmt(after_score),
            _delta_fmt(delta),
            reason,
        ])

    cipher_name = _escape_latex(history.cipher_name or "cipher")
    return _format_table(
        headers, rows,
        caption=f"Iteration history for {cipher_name}: LLM-guided improvement trajectory",
        label="iteration_summary",
        column_format="c l l c c c c l",
    )


# ---------------------------------------------------------------------------
# Table 2: Detailed metric comparison (accepted iterations only)
# ---------------------------------------------------------------------------

def accepted_metrics_table(history: IterationHistory) -> str:
    """Generate a LaTeX table showing metric deltas for accepted iterations only."""
    accepted = history.accepted()
    if not accepted:
        return "% No accepted iterations to report."

    headers = [
        "\\#",
        "$\\Delta$ PT Aval.",
        "$\\Delta$ Key Aval.",
        "$\\Delta$ Overall",
        "$\\Delta$ SAC PT",
        "$\\Delta$ SAC Key",
        "Patch Summary",
    ]

    rows = []
    for rec in accepted:
        d = rec.metric_deltas or {}
        summary = _escape_latex(rec.patch_summary[:35]) if rec.patch_summary else "---"

        rows.append([
            str(rec.iteration_id),
            _delta_fmt(d.get("pt_avalanche_score")),
            _delta_fmt(d.get("key_avalanche_score")),
            _delta_fmt(d.get("overall_score")),
            _delta_fmt(d.get("sac_deviation_pt")),
            _delta_fmt(d.get("sac_deviation_key")),
            summary,
        ])

    cipher_name = _escape_latex(history.cipher_name or "cipher")
    return _format_table(
        headers, rows,
        caption=f"Accepted improvement deltas for {cipher_name}",
        label="accepted_metrics",
        column_format="c c c c c c l",
    )


# ---------------------------------------------------------------------------
# Table 3: Summary statistics
# ---------------------------------------------------------------------------

def summary_statistics_table(history: IterationHistory) -> str:
    """Generate a small summary table with aggregate statistics."""
    if not history.records:
        return "% No iterations to report."

    total = history.count
    n_accepted = len(history.accepted())
    n_rejected = len(history.rejected())
    accept_rate = n_accepted / total if total > 0 else 0.0

    # Compute cumulative improvement
    accepted = history.accepted()
    cumulative_overall = 0.0
    best_single_delta = 0.0
    for rec in accepted:
        d = (rec.metric_deltas or {}).get("overall_score")
        if d is not None:
            cumulative_overall += d
            if abs(d) > abs(best_single_delta):
                best_single_delta = d

    # Models used
    models = set(rec.model_used for rec in history.records if rec.model_used)
    models_str = ", ".join(_escape_latex(m.split("/")[-1]) for m in sorted(models))

    headers = ["Statistic", "Value"]
    rows = [
        ["Total iterations", str(total)],
        ["Accepted", str(n_accepted)],
        ["Rejected", str(n_rejected)],
        ["Accept rate", f"${accept_rate:.1%}$".replace("%", "\\%")],
        ["Cumulative $\\Delta$ overall score", _delta_fmt(cumulative_overall)],
        ["Best single $\\Delta$ overall score", _delta_fmt(best_single_delta)],
        ["Models used", models_str if models_str else "---"],
    ]

    cipher_name = _escape_latex(history.cipher_name or "cipher")
    return _format_table(
        headers, rows,
        caption=f"Iterative improvement summary for {cipher_name}",
        label="iteration_stats",
        column_format="l r",
    )


# ---------------------------------------------------------------------------
# Export all tables
# ---------------------------------------------------------------------------

def export_iteration_tables(
    history: IterationHistory,
    output_dir: str | Path,
) -> List[str]:
    """Generate all iteration LaTeX tables and write .tex files.

    Args:
        history: The iteration history to export.
        output_dir: Directory to write .tex files.

    Returns:
        List of file paths that were created.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    tables = {
        "iteration_summary.tex": iteration_summary_table(history),
        "accepted_metrics.tex": accepted_metrics_table(history),
        "iteration_stats.tex": summary_statistics_table(history),
    }

    paths: List[str] = []
    for filename, latex in tables.items():
        path = out / filename
        path.write_text(latex, encoding="utf-8")
        paths.append(str(path))

    return paths
