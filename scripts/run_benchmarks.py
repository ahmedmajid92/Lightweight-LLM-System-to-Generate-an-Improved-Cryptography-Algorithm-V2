"""CLI entry point for Phase 5 benchmarking.

Usage:
    python scripts/run_benchmarks.py                                    # full suite
    python scripts/run_benchmarks.py --algorithms AES SPECK DES         # subset
    python scripts/run_benchmarks.py --max-iterations 3 --reps 1        # quick test
    python scripts/run_benchmarks.py --latex-only benchmarks/.../results.json

Research / education only. Do NOT use in production.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Ensure project root is on path
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from cipherlab.config import load_settings
from cipherlab.utils.repro import write_json, utc_timestamp
from cipherlab.evaluation.benchmark_runner import (
    build_default_suite,
    run_benchmark_suite,
    BenchmarkSuiteResult,
    ExperimentResult,
    IterationMetrics,
)
from cipherlab.evaluation.latex_exporter import LaTeXExporter
from cipherlab.evaluation.dataset_exporter import export_dataset_jsonl, export_full_jsonl


def _cli_progress(message: str, current: int, total: int) -> None:
    """Print progress to stderr."""
    pct = (current / total * 100) if total > 0 else 0
    print(f"  [{current + 1}/{total}] ({pct:.0f}%) {message}", file=sys.stderr)


def _reconstruct_suite_result(data: dict) -> BenchmarkSuiteResult:
    """Reconstruct a BenchmarkSuiteResult from a saved JSON dict."""
    experiments = []
    for exp_data in data.get("experiments", []):
        history = []
        for m in exp_data.pop("iteration_history", []):
            history.append(IterationMetrics(**m))
        exp = ExperimentResult(**exp_data, iteration_history=history)
        experiments.append(exp)

    return BenchmarkSuiteResult(
        suite_name=data.get("suite_name", ""),
        timestamp=data.get("timestamp", ""),
        total_experiments=data.get("total_experiments", len(experiments)),
        completed=data.get("completed", 0),
        failed=data.get("failed", 0),
        experiments=experiments,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 5 Benchmark Runner - Empirical Model Comparison",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/run_benchmarks.py --algorithms AES --reps 1    # quick test\n"
            "  python scripts/run_benchmarks.py                               # full suite\n"
            "  python scripts/run_benchmarks.py --latex-only results.json     # regenerate tables\n"
        ),
    )

    parser.add_argument(
        "--algorithms", nargs="+", default=None,
        help="Algorithm names to benchmark (default: all 12)",
    )
    parser.add_argument(
        "--max-iterations", type=int, default=5,
        help="Max feedback->evolve cycles per experiment (default: 5)",
    )
    parser.add_argument(
        "--reps", type=int, default=3,
        help="Repetitions per (algorithm, model) pair (default: 3)",
    )
    parser.add_argument(
        "--seed", type=int, default=1337,
        help="Base random seed (default: 1337)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="benchmarks",
        help="Output directory (default: benchmarks)",
    )
    parser.add_argument(
        "--sac-trials", type=int, default=500,
        help="SAC trials per input bit (default: 500)",
    )
    parser.add_argument(
        "--roundtrip-vectors", type=int, default=1000,
        help="Roundtrip test vectors per evaluation (default: 1000)",
    )
    parser.add_argument(
        "--latex-only", type=str, default=None, metavar="RESULTS_JSON",
        help="Path to existing results.json - regenerate LaTeX tables only",
    )
    parser.add_argument(
        "--skip-latex", action="store_true",
        help="Skip LaTeX table generation",
    )
    parser.add_argument(
        "--skip-jsonl", action="store_true",
        help="Skip JSONL dataset generation",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # ------------------------------------------------------------------
    # LaTeX-only mode: regenerate tables from existing results
    # ------------------------------------------------------------------
    if args.latex_only:
        print(f"Loading results from {args.latex_only}...")
        with open(args.latex_only, "r", encoding="utf-8") as f:
            data = json.load(f)

        suite_result = _reconstruct_suite_result(data)
        out_dir = Path(args.latex_only).parent / "tables"

        exporter = LaTeXExporter(suite_result)
        paths = exporter.export_all(str(out_dir))
        print(f"Generated {len(paths)} LaTeX tables in {out_dir}:")
        for p in paths:
            print(f"  {p}")
        return

    # ------------------------------------------------------------------
    # Full benchmark run
    # ------------------------------------------------------------------
    settings = load_settings()

    # Check API keys
    if not settings.openai_api_key:
        print("WARNING: OPENAI_API_KEY not set. OpenAI experiments will fail.",
              file=sys.stderr)
    if not settings.openrouter_api_key:
        print("WARNING: OPENROUTER_API_KEY not set. DeepSeek experiments will fail.",
              file=sys.stderr)

    suite = build_default_suite(
        settings,
        algorithms=args.algorithms,
        max_iterations=args.max_iterations,
        seed=args.seed,
        repetitions=args.reps,
        num_roundtrip_vectors=args.roundtrip_vectors,
        sac_trials=args.sac_trials,
    )

    total = len(suite.experiments)
    print(f"Benchmark suite: {total} experiments")
    print(f"  Algorithms: {len(set(e.algorithm for e in suite.experiments))}")
    print(f"  Models: {len(set(e.model.label for e in suite.experiments))}")
    print(f"  Repetitions: {args.reps}")
    print(f"  Max iterations per experiment: {args.max_iterations}")
    print()

    # Run
    suite_result = run_benchmark_suite(
        settings, suite, progress_callback=_cli_progress,
    )

    out_dir = Path(suite.output_dir) / suite_result.timestamp
    print(f"\nBenchmark complete: {suite_result.completed}/{suite_result.total_experiments} "
          f"succeeded, {suite_result.failed} failed.")

    # LaTeX tables
    if not args.skip_latex:
        exporter = LaTeXExporter(suite_result)
        paths = exporter.export_all(str(out_dir / "tables"))
        print(f"Generated {len(paths)} LaTeX tables in {out_dir / 'tables'}")

    # JSONL dataset
    if not args.skip_jsonl:
        compact_path = export_dataset_jsonl(
            suite_result.experiments, out_dir / "dataset.jsonl",
        )
        full_path = export_full_jsonl(
            suite_result.experiments, out_dir / "dataset_full.jsonl",
        )
        print(f"Dataset exported:")
        print(f"  Compact: {compact_path}")
        print(f"  Full:    {full_path}")

    print(f"\nAll results saved to: {out_dir}")


if __name__ == "__main__":
    main()
