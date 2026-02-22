"""Automated LaTeX table generation for thesis data presentation.

Converts benchmark JSON results into publication-ready LaTeX tables
with booktabs formatting, mean ± std aggregation, and proper escaping.

Research / education only. Do NOT use in production.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from .benchmark_runner import BenchmarkSuiteResult, ExperimentResult


# ---------------------------------------------------------------------------
# Data structure
# ---------------------------------------------------------------------------

@dataclass
class LaTeXTable:
    """A generated LaTeX table."""
    name: str       # e.g., "model_comparison"
    caption: str    # LaTeX \\caption text
    label: str      # LaTeX \\label, e.g., "tab:model_comparison"
    latex: str      # Complete LaTeX source


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mean_pm_std(values: List[float], fmt: str = ".3f") -> str:
    """Format as 'mean ± std' using LaTeX \\pm notation.

    Returns '$mean \\pm std$' or just '$value$' if single value.
    """
    values = [v for v in values if v is not None]
    if not values:
        return "---"
    m = statistics.mean(values)
    if len(values) == 1:
        return f"${m:{fmt}}$"
    s = statistics.stdev(values)
    return f"${m:{fmt}} \\pm {s:{fmt}}$"


def _mean_pm_std_int(values: List[Optional[int]], default: str = "---") -> str:
    """Format integer values as 'mean ± std'."""
    clean = [float(v) for v in values if v is not None]
    if not clean:
        return default
    m = statistics.mean(clean)
    if len(clean) == 1:
        return f"${m:.1f}$"
    s = statistics.stdev(clean)
    return f"${m:.1f} \\pm {s:.1f}$"


def _escape_latex(text: str) -> str:
    """Escape special LaTeX characters in text."""
    for ch in ["_", "&", "%", "#", "$", "{", "}"]:
        text = text.replace(ch, f"\\{ch}")
    return text


def _format_table(
    headers: List[str],
    rows: List[List[str]],
    caption: str,
    label: str,
    column_format: Optional[str] = None,
) -> str:
    """Generate LaTeX tabular with booktabs formatting.

    Args:
        headers: Column header strings.
        rows: List of rows, each a list of cell strings.
        caption: Table caption.
        label: Table label (without 'tab:' prefix).
        column_format: LaTeX column spec. Defaults to 'l' + 'c' * (ncols-1).

    Returns:
        Complete LaTeX table source.
    """
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
# LaTeX exporter class
# ---------------------------------------------------------------------------

class LaTeXExporter:
    """Generates publication-ready LaTeX tables from benchmark results."""

    def __init__(self, suite_result: BenchmarkSuiteResult):
        self._suite = suite_result
        self._experiments = suite_result.experiments

    def _group_by(self, key: str) -> Dict[str, List[ExperimentResult]]:
        """Group experiments by a field name."""
        groups: Dict[str, List[ExperimentResult]] = {}
        for exp in self._experiments:
            val = getattr(exp, key, "unknown")
            groups.setdefault(val, []).append(exp)
        return groups

    # -----------------------------------------------------------------------
    # Table 1: Model Performance Comparison
    # -----------------------------------------------------------------------

    def model_comparison_table(self) -> LaTeXTable:
        """Rows = models, Cols = aggregated metrics."""
        groups = self._group_by("model_label")

        headers = [
            "Model",
            "Iters to RT",
            "SAC Dev (PT)",
            "SAC Dev (Key)",
            "Evolutions",
            "Time (s)",
        ]

        rows = []
        for label in sorted(groups.keys()):
            exps = groups[label]
            rt_iters = [e.iterations_to_roundtrip_pass for e in exps]
            sac_pt = [e.final_sac_pt_deviation for e in exps]
            sac_key = [e.final_sac_key_deviation for e in exps]
            evos = [e.total_evolutions_succeeded for e in exps]
            times = [e.total_time_seconds for e in exps]

            rows.append([
                _escape_latex(label),
                _mean_pm_std_int(rt_iters),
                _mean_pm_std(sac_pt),
                _mean_pm_std(sac_key),
                _mean_pm_std([float(e) for e in evos], fmt=".1f"),
                _mean_pm_std(times, fmt=".1f"),
            ])

        latex = _format_table(
            headers, rows,
            caption="Model Performance Comparison for Cipher Improvement",
            label="model_comparison",
        )

        return LaTeXTable(
            name="model_comparison",
            caption="Model Performance Comparison for Cipher Improvement",
            label="tab:model_comparison",
            latex=latex,
        )

    # -----------------------------------------------------------------------
    # Table 2: Architecture Difficulty Comparison
    # -----------------------------------------------------------------------

    def architecture_comparison_table(self) -> LaTeXTable:
        """Rows = architectures (SPN, Feistel, ARX), Cols = metrics."""
        groups = self._group_by("architecture")

        headers = [
            "Architecture",
            "Iters to RT",
            "SAC Dev (PT)",
            "SAC Dev (Key)",
            "Evolutions",
            "Time (s)",
        ]

        rows = []
        for arch in ["SPN", "FEISTEL", "ARX"]:
            exps = groups.get(arch, [])
            if not exps:
                continue
            rt_iters = [e.iterations_to_roundtrip_pass for e in exps]
            sac_pt = [e.final_sac_pt_deviation for e in exps]
            sac_key = [e.final_sac_key_deviation for e in exps]
            evos = [e.total_evolutions_succeeded for e in exps]
            times = [e.total_time_seconds for e in exps]

            rows.append([
                arch,
                _mean_pm_std_int(rt_iters),
                _mean_pm_std(sac_pt),
                _mean_pm_std(sac_key),
                _mean_pm_std([float(e) for e in evos], fmt=".1f"),
                _mean_pm_std(times, fmt=".1f"),
            ])

        latex = _format_table(
            headers, rows,
            caption="Architecture Difficulty Comparison for LLM-Based Cipher Improvement",
            label="architecture_comparison",
        )

        return LaTeXTable(
            name="architecture_comparison",
            caption="Architecture Difficulty Comparison for LLM-Based Cipher Improvement",
            label="tab:architecture_comparison",
            latex=latex,
        )

    # -----------------------------------------------------------------------
    # Table 3: Per-Algorithm Best Results
    # -----------------------------------------------------------------------

    def algorithm_detail_table(self) -> LaTeXTable:
        """Rows = algorithms, Cols = best model + metrics."""
        groups = self._group_by("algorithm")

        headers = [
            "Algorithm",
            "Arch.",
            "Best Model",
            "Iters to RT",
            "Final SAC Dev",
            "Evolutions",
            "Time (s)",
        ]

        rows = []
        for algo in sorted(groups.keys()):
            exps = groups[algo]
            # Best = lowest final SAC deviation (PT)
            best = min(exps, key=lambda e: e.final_sac_pt_deviation)
            rows.append([
                _escape_latex(algo),
                best.architecture,
                _escape_latex(best.model_label),
                str(best.iterations_to_roundtrip_pass) if best.iterations_to_roundtrip_pass is not None else "---",
                f"${best.final_sac_pt_deviation:.4f}$",
                str(best.total_evolutions_succeeded),
                f"${best.total_time_seconds:.1f}$",
            ])

        latex = _format_table(
            headers, rows,
            caption="Per-Algorithm Best Results Across All Models",
            label="algorithm_detail",
        )

        return LaTeXTable(
            name="algorithm_detail",
            caption="Per-Algorithm Best Results Across All Models",
            label="tab:algorithm_detail",
            latex=latex,
        )

    # -----------------------------------------------------------------------
    # Table 4: Iteration Convergence for a Specific Algorithm
    # -----------------------------------------------------------------------

    def iteration_convergence_table(self, algorithm: str) -> LaTeXTable:
        """Convergence curve: rows = iterations, cols = SAC dev per model."""
        # Group experiments by model for this algorithm
        model_groups: Dict[str, List[ExperimentResult]] = {}
        for exp in self._experiments:
            if exp.algorithm == algorithm:
                model_groups.setdefault(exp.model_label, []).append(exp)

        model_labels = sorted(model_groups.keys())
        if not model_labels:
            return LaTeXTable(
                name=f"convergence_{algorithm}",
                caption=f"No data for {algorithm}",
                label=f"tab:convergence_{algorithm.lower()}",
                latex="% No data available",
            )

        # Find max iterations across all experiments for this algorithm
        max_iters = max(
            len(exp.iteration_history)
            for exps in model_groups.values()
            for exp in exps
        )

        headers = ["Iteration"] + [_escape_latex(ml) for ml in model_labels]

        rows = []
        for i in range(max_iters):
            row = [str(i)]
            for ml in model_labels:
                exps = model_groups[ml]
                devs = []
                for exp in exps:
                    if i < len(exp.iteration_history):
                        devs.append(exp.iteration_history[i].sac_pt_deviation)
                if devs:
                    row.append(_mean_pm_std(devs, fmt=".4f"))
                else:
                    row.append("---")
            rows.append(row)

        algo_escaped = _escape_latex(algorithm)
        latex = _format_table(
            headers, rows,
            caption=f"SAC Deviation Convergence for {algo_escaped} by Model",
            label=f"convergence_{algorithm.lower()}",
        )

        return LaTeXTable(
            name=f"convergence_{algorithm}",
            caption=f"SAC Deviation Convergence for {algo_escaped} by Model",
            label=f"tab:convergence_{algorithm.lower()}",
            latex=latex,
        )

    # -----------------------------------------------------------------------
    # Table 5: Token/Cost Comparison
    # -----------------------------------------------------------------------

    def token_cost_table(self) -> LaTeXTable:
        """Rows = models, Cols = token usage and timing."""
        groups = self._group_by("model_label")

        headers = [
            "Model",
            "Avg Input Tokens",
            "Avg Output Tokens",
            "Total Tokens",
            "Avg Time/Iter (s)",
        ]

        rows = []
        for label in sorted(groups.keys()):
            exps = groups[label]
            inp = [float(e.total_tokens_input) for e in exps]
            out = [float(e.total_tokens_output) for e in exps]
            total = [float(e.total_tokens_input + e.total_tokens_output) for e in exps]
            # Average time per iteration
            time_per_iter = []
            for e in exps:
                if e.iterations_run > 0:
                    time_per_iter.append(e.total_time_seconds / e.iterations_run)

            rows.append([
                _escape_latex(label),
                _mean_pm_std(inp, fmt=".0f"),
                _mean_pm_std(out, fmt=".0f"),
                _mean_pm_std(total, fmt=".0f"),
                _mean_pm_std(time_per_iter, fmt=".1f") if time_per_iter else "---",
            ])

        latex = _format_table(
            headers, rows,
            caption="Token Usage and Cost Comparison by Model",
            label="token_cost",
        )

        return LaTeXTable(
            name="token_cost",
            caption="Token Usage and Cost Comparison by Model",
            label="tab:token_cost",
            latex=latex,
        )

    # -----------------------------------------------------------------------
    # Export all tables
    # -----------------------------------------------------------------------

    def export_all(self, output_dir: str) -> List[str]:
        """Generate all tables and write .tex files.

        Args:
            output_dir: Directory to write .tex files.

        Returns:
            List of file paths that were created.
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        tables: List[LaTeXTable] = [
            self.model_comparison_table(),
            self.architecture_comparison_table(),
            self.algorithm_detail_table(),
            self.token_cost_table(),
        ]

        # Add per-algorithm convergence tables
        algorithms = sorted(set(e.algorithm for e in self._experiments))
        for algo in algorithms:
            tables.append(self.iteration_convergence_table(algo))

        paths: List[str] = []
        for table in tables:
            path = out / f"{table.name}.tex"
            path.write_text(table.latex, encoding="utf-8")
            paths.append(str(path))

        return paths
