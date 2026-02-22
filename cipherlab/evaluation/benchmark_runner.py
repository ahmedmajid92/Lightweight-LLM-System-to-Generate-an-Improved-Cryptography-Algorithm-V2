"""Automated benchmarking orchestrator for empirical model comparison.

Systematically runs the full generation→evaluation→feedback→evolution loop
across multiple LLM models (OpenAI vs DeepSeek-V3 vs DeepSeek-R1) and
algorithms, tracking quantitative metrics for thesis data generation.

Research / education only. Do NOT use in production.
"""
from __future__ import annotations

import copy
import inspect
import logging
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field

from cipherlab.config import Settings
from cipherlab.utils.repro import write_json, utc_timestamp

import sys
_project_root = Path(__file__).parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from AlgorithmsBlock import (
    CipherSpec as ABSpec,
    build_cipher as ab_build_cipher,
    get_template,
    list_algorithms,
    ComponentRegistry,
)

from cipherlab.cipher.spec import CipherSpec as PydanticSpec
from cipherlab.evaluation.roundtrip import run_roundtrip_tests
from cipherlab.evaluation.avalanche import compute_sac
from cipherlab.evaluation.sbox_analysis import analyze_all_sboxes
from cipherlab.evaluation.report import EvaluationReport
from cipherlab.evaluation.feedback import run_feedback_cycle
from cipherlab.evolution.ast_analyzer import detect_mismatches
from cipherlab.evolution.dynamic_loader import evolve_all_mismatches

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration data structures
# ---------------------------------------------------------------------------

class ModelConfig(BaseModel):
    """Which LLM to use for an experiment."""
    label: str                                    # Human-readable: "DeepSeek-R1"
    provider: Literal["openai", "openrouter"]     # Route selector
    model_id: str                                 # API model ID


DEFAULT_MODELS: List[ModelConfig] = [
    ModelConfig(label="GPT-5.2", provider="openai", model_id="gpt-5.2"),
    ModelConfig(
        label="DeepSeek-V3",
        provider="openrouter",
        model_id="deepseek/deepseek-chat-v3-0324",
    ),
    ModelConfig(
        label="DeepSeek-R1",
        provider="openrouter",
        model_id="deepseek/deepseek-r1",
    ),
]


class ExperimentConfig(BaseModel):
    """Configuration for a single benchmarking experiment."""
    experiment_id: str
    algorithm: str
    model: ModelConfig
    max_iterations: int = Field(default=5, ge=1, le=50)
    seed: int = Field(default=1337)
    num_roundtrip_vectors: int = Field(default=1000, ge=10)
    sac_trials: int = Field(default=500, ge=10)


class BenchmarkSuiteConfig(BaseModel):
    """Full benchmark suite definition."""
    suite_name: str = "phase5_benchmark"
    experiments: List[ExperimentConfig]
    output_dir: str = "benchmarks"


# ---------------------------------------------------------------------------
# Metrics data structures
# ---------------------------------------------------------------------------

@dataclass
class IterationMetrics:
    """Metrics captured at each iteration of the improvement loop."""
    iteration: int
    # Roundtrip
    roundtrip_passed: bool = False
    roundtrip_success_rate: float = 0.0
    roundtrip_elapsed_seconds: float = 0.0
    # SAC (plaintext)
    sac_pt_mean: float = 0.0
    sac_pt_deviation: float = 0.0
    sac_pt_passes: bool = False
    # SAC (key)
    sac_key_mean: float = 0.0
    sac_key_deviation: float = 0.0
    sac_key_passes: bool = False
    # S-box
    sbox_ddt_max: Optional[int] = None
    sbox_lat_max: Optional[int] = None
    # Evolution
    evolutions_attempted: int = 0
    evolutions_succeeded: int = 0
    # LLM metadata
    feedback_model_used: str = ""
    mutation_model_used: str = ""
    reasoning_trace: Optional[str] = None
    patch_summary: str = ""
    patch_replace_components: Optional[Dict[str, str]] = None
    patch_new_rounds: Optional[int] = None
    # Token usage
    token_usage_input: int = 0
    token_usage_output: int = 0
    # Error
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ExperimentResult:
    """Complete result of one experiment (algorithm + model + N iterations)."""
    experiment_id: str
    algorithm: str
    architecture: str
    model_label: str
    model_id: str
    seed: int
    # Aggregate metrics
    iterations_run: int = 0
    iterations_to_roundtrip_pass: Optional[int] = None
    final_sac_pt_deviation: float = 0.0
    final_sac_pt_mean: float = 0.0
    final_sac_key_deviation: float = 0.0
    final_sac_key_mean: float = 0.0
    total_evolutions_attempted: int = 0
    total_evolutions_succeeded: int = 0
    total_time_seconds: float = 0.0
    total_tokens_input: int = 0
    total_tokens_output: int = 0
    # Per-iteration detail
    iteration_history: List[IterationMetrics] = field(default_factory=list)
    # Final state capture
    final_spec: Optional[Dict[str, Any]] = None
    final_components: Optional[Dict[str, str]] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["iteration_history"] = [m.to_dict() for m in self.iteration_history]
        return d


@dataclass
class BenchmarkSuiteResult:
    """All experiments in a suite."""
    suite_name: str
    timestamp: str
    total_experiments: int = 0
    completed: int = 0
    failed: int = 0
    experiments: List[ExperimentResult] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "suite_name": self.suite_name,
            "timestamp": self.timestamp,
            "total_experiments": self.total_experiments,
            "completed": self.completed,
            "failed": self.failed,
            "experiments": [e.to_dict() for e in self.experiments],
        }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _make_settings_for_model(base: Settings, model_config: ModelConfig) -> Settings:
    """Create a Settings copy that forces routing to a specific model.

    For OpenAI: disables OpenRouter by nullifying the API key.
    For OpenRouter: sets the reasoning/fast model to the target model ID.
    """
    s = base.model_copy(deep=True)
    if model_config.provider == "openai":
        s.openrouter_api_key = None
        s.openai_model_quality = model_config.model_id
        s.openai_model_code = model_config.model_id
    elif model_config.provider == "openrouter":
        s.openrouter_model_reasoning = model_config.model_id
        s.openrouter_model_fast = model_config.model_id
    return s


def _to_pydantic(spec: ABSpec) -> PydanticSpec:
    """Convert AlgorithmsBlock CipherSpec to Pydantic CipherSpec for feedback."""
    return PydanticSpec(
        name=spec.name,
        architecture=spec.architecture,
        block_size_bits=spec.block_size_bits,
        key_size_bits=spec.key_size_bits,
        rounds=spec.rounds,
        components=dict(spec.components),
        seed=spec.seed,
        notes=spec.notes,
    )


def _extract_token_usage(raw_response: Any) -> Tuple[int, int]:
    """Extract (input_tokens, output_tokens) from a raw API response.

    Handles both OpenAI Responses API and OpenRouter Chat Completions API.
    """
    usage = getattr(raw_response, "usage", None)
    if usage is None:
        return 0, 0
    inp = getattr(usage, "input_tokens", 0) or getattr(usage, "prompt_tokens", 0) or 0
    out = getattr(usage, "output_tokens", 0) or getattr(usage, "completion_tokens", 0) or 0
    return inp, out


def _extract_evolved_sources(registry: ComponentRegistry) -> Dict[str, str]:
    """Get source code of any evolved components from the registry."""
    result: Dict[str, str] = {}
    for cid in registry.list_ids():
        if "_evolved_" in cid:
            comp = registry.get(cid)
            try:
                result[cid] = inspect.getsource(comp.forward)
            except (OSError, TypeError):
                result[cid] = "<source unavailable>"
    return result


def _spec_to_dict(spec: ABSpec) -> Dict[str, Any]:
    """Serialize an AlgorithmsBlock CipherSpec to a dict."""
    return {
        "name": spec.name,
        "architecture": spec.architecture,
        "block_size_bits": spec.block_size_bits,
        "key_size_bits": spec.key_size_bits,
        "rounds": spec.rounds,
        "components": dict(spec.components),
        "version": spec.version,
        "notes": spec.notes,
        "seed": spec.seed,
        "word_size": getattr(spec, "word_size", 32),
    }


def _slug(text: str) -> str:
    """Convert to filesystem-safe slug."""
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in text)


# ---------------------------------------------------------------------------
# Suite builder
# ---------------------------------------------------------------------------

def build_default_suite(
    settings: Settings,
    algorithms: Optional[List[str]] = None,
    models: Optional[List[ModelConfig]] = None,
    max_iterations: int = 5,
    seed: int = 1337,
    repetitions: int = 3,
    num_roundtrip_vectors: int = 1000,
    sac_trials: int = 500,
) -> BenchmarkSuiteConfig:
    """Generate a standard benchmark suite config.

    Creates one ExperimentConfig per (algorithm, model, repetition).
    Seeds increment per repetition: seed, seed+1, seed+2, ...

    Args:
        settings: Application settings (used to detect available APIs).
        algorithms: Algorithm names to benchmark. None = all 12.
        models: Model configs to compare. None = GPT-5.2 + DeepSeek-V3 + DeepSeek-R1.
        max_iterations: Max feedback→evolve cycles per experiment.
        seed: Base random seed.
        repetitions: Number of repetitions per (algorithm, model) pair.
        num_roundtrip_vectors: Test vectors for roundtrip verification.
        sac_trials: Trials per input bit for SAC computation.

    Returns:
        BenchmarkSuiteConfig with all experiment definitions.
    """
    if algorithms is None:
        algorithms = list_algorithms()
    if models is None:
        models = list(DEFAULT_MODELS)

    experiments: List[ExperimentConfig] = []

    for algo in sorted(algorithms):
        for model in models:
            for rep in range(repetitions):
                exp_id = f"{_slug(algo)}_{_slug(model.label)}_run{rep}"
                experiments.append(ExperimentConfig(
                    experiment_id=exp_id,
                    algorithm=algo,
                    model=model,
                    max_iterations=max_iterations,
                    seed=seed + rep,
                    num_roundtrip_vectors=num_roundtrip_vectors,
                    sac_trials=sac_trials,
                ))

    return BenchmarkSuiteConfig(
        suite_name="phase5_benchmark",
        experiments=experiments,
    )


# ---------------------------------------------------------------------------
# Single experiment runner
# ---------------------------------------------------------------------------

def run_single_experiment(
    settings: Settings,
    config: ExperimentConfig,
    progress_callback: Optional[Callable[[str, int, int], None]] = None,
) -> ExperimentResult:
    """Run one complete experiment: template → [evaluate → feedback → patch → evolve] × N.

    Each experiment uses a FRESH ComponentRegistry to prevent cross-experiment leakage.
    Exceptions per iteration are caught and recorded in IterationMetrics.error.

    Args:
        settings: Base settings (will be overridden per model).
        config: Experiment configuration.
        progress_callback: Optional callback(message, current, total).

    Returns:
        ExperimentResult with per-iteration metrics and final state.
    """
    forced_settings = _make_settings_for_model(settings, config.model)

    # Fresh registry per experiment
    registry = ComponentRegistry()

    # Get algorithm template
    spec = get_template(config.algorithm, seed=config.seed)

    result = ExperimentResult(
        experiment_id=config.experiment_id,
        algorithm=config.algorithm,
        architecture=spec.architecture,
        model_label=config.model.label,
        model_id=config.model.model_id,
        seed=config.seed,
    )

    start_time = time.perf_counter()

    for iteration in range(config.max_iterations):
        if progress_callback:
            progress_callback(
                f"{config.experiment_id} iter {iteration}",
                iteration,
                config.max_iterations,
            )

        metrics = IterationMetrics(iteration=iteration)

        try:
            # Step 1: Build cipher
            cipher = ab_build_cipher(spec, registry)

            # Step 2: Roundtrip test
            rt = run_roundtrip_tests(
                spec,
                num_vectors=config.num_roundtrip_vectors,
                seed=config.seed,
                registry=registry,
            )
            metrics.roundtrip_passed = rt.is_perfect
            metrics.roundtrip_success_rate = rt.success_rate
            metrics.roundtrip_elapsed_seconds = rt.elapsed_seconds

            if rt.is_perfect and result.iterations_to_roundtrip_pass is None:
                result.iterations_to_roundtrip_pass = iteration

            # Step 3: SAC (plaintext)
            sac_pt = compute_sac(
                cipher,
                block_size_bits=spec.block_size_bits,
                key_size_bits=spec.key_size_bits,
                input_type="plaintext",
                trials=config.sac_trials,
                seed=config.seed,
                algorithm_name=spec.name,
                architecture=spec.architecture,
            )
            metrics.sac_pt_mean = sac_pt.global_mean
            metrics.sac_pt_deviation = sac_pt.sac_deviation
            metrics.sac_pt_passes = sac_pt.passes_sac

            # Step 4: SAC (key)
            sac_key = compute_sac(
                cipher,
                block_size_bits=spec.block_size_bits,
                key_size_bits=spec.key_size_bits,
                input_type="key",
                trials=config.sac_trials,
                seed=config.seed,
                algorithm_name=spec.name,
                architecture=spec.architecture,
            )
            metrics.sac_key_mean = sac_key.global_mean
            metrics.sac_key_deviation = sac_key.sac_deviation
            metrics.sac_key_passes = sac_key.passes_sac

            # Step 5: S-box analysis
            sbox_results = analyze_all_sboxes(registry)
            if sbox_results:
                metrics.sbox_ddt_max = max(sb.ddt_max for sb in sbox_results)
                metrics.sbox_lat_max = max(sb.lat_max_abs for sb in sbox_results)

            # Early exit: everything passes
            if rt.is_perfect and sac_pt.passes_sac and sac_key.passes_sac:
                result.iteration_history.append(metrics)
                break

            # Step 6: Build evaluation report and run feedback cycle
            report = EvaluationReport(
                roundtrip_results=[rt],
                sac_results=[sac_pt, sac_key],
                sbox_results=sbox_results,
            )

            pydantic_spec = _to_pydantic(spec)
            fb = run_feedback_cycle(forced_settings, pydantic_spec, report)

            metrics.feedback_model_used = fb.model_used
            metrics.reasoning_trace = fb.reasoning_trace

            # Extract token usage
            inp_tok, out_tok = _extract_token_usage(fb.raw_response)
            metrics.token_usage_input = inp_tok
            metrics.token_usage_output = out_tok

            # Step 7: Apply patch
            if fb.patch:
                metrics.patch_summary = fb.patch.summary
                if fb.patch.new_rounds is not None:
                    spec.rounds = fb.patch.new_rounds
                    metrics.patch_new_rounds = fb.patch.new_rounds
                if fb.patch.replace_components:
                    spec.components.update(fb.patch.replace_components)
                    metrics.patch_replace_components = dict(fb.patch.replace_components)
                if fb.patch.add_notes:
                    spec.notes = (spec.notes + "\n" + fb.patch.add_notes).strip()

            # Step 8: Detect mismatches and evolve
            pydantic_spec_updated = _to_pydantic(spec)
            mismatches = detect_mismatches(pydantic_spec_updated, registry)
            blocking = [m for m in mismatches if m.severity == "blocking"]
            if blocking:
                evo = evolve_all_mismatches(
                    forced_settings, pydantic_spec_updated, registry,
                )
                metrics.evolutions_attempted = evo.evolutions_attempted
                metrics.evolutions_succeeded = evo.evolutions_succeeded
                # Update spec with evolved component IDs
                if evo.all_resolved():
                    for loaded in evo.evolved_components:
                        for role, cid in spec.components.items():
                            orig = loaded.component_id.rsplit("_evolved_", 1)[0]
                            if cid == orig:
                                spec.components[role] = loaded.component_id

        except Exception as e:
            metrics.error = f"{type(e).__name__}: {str(e)}"
            logger.warning(
                "Experiment %s iteration %d failed: %s",
                config.experiment_id, iteration, metrics.error,
            )

        result.iteration_history.append(metrics)

    # Finalize result
    elapsed = time.perf_counter() - start_time
    result.iterations_run = len(result.iteration_history)
    result.total_time_seconds = round(elapsed, 2)

    # Aggregate from last successful iteration
    if result.iteration_history:
        last = result.iteration_history[-1]
        result.final_sac_pt_deviation = last.sac_pt_deviation
        result.final_sac_pt_mean = last.sac_pt_mean
        result.final_sac_key_deviation = last.sac_key_deviation
        result.final_sac_key_mean = last.sac_key_mean

    # Sum evolution counts and token usage
    result.total_evolutions_attempted = sum(
        m.evolutions_attempted for m in result.iteration_history
    )
    result.total_evolutions_succeeded = sum(
        m.evolutions_succeeded for m in result.iteration_history
    )
    result.total_tokens_input = sum(
        m.token_usage_input for m in result.iteration_history
    )
    result.total_tokens_output = sum(
        m.token_usage_output for m in result.iteration_history
    )

    # Capture final state
    result.final_spec = _spec_to_dict(spec)
    result.final_components = _extract_evolved_sources(registry)

    return result


# ---------------------------------------------------------------------------
# Suite runner
# ---------------------------------------------------------------------------

def run_benchmark_suite(
    settings: Settings,
    suite: BenchmarkSuiteConfig,
    progress_callback: Optional[Callable[[str, int, int], None]] = None,
) -> BenchmarkSuiteResult:
    """Run all experiments in a suite sequentially.

    Saves a checkpoint JSON after each experiment for crash resilience.

    Args:
        settings: Base settings with API keys.
        suite: Suite configuration with all experiments.
        progress_callback: Optional callback(experiment_id, current, total).

    Returns:
        BenchmarkSuiteResult with all experiment results.
    """
    suite_result = BenchmarkSuiteResult(
        suite_name=suite.suite_name,
        timestamp=utc_timestamp(),
        total_experiments=len(suite.experiments),
    )

    # Create output directory for checkpoints
    out_dir = Path(suite.output_dir) / suite_result.timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_dir / "_checkpoint.json"

    for idx, exp_config in enumerate(suite.experiments):
        if progress_callback:
            progress_callback(
                exp_config.experiment_id, idx, len(suite.experiments),
            )

        logger.info(
            "Starting experiment %d/%d: %s",
            idx + 1, len(suite.experiments), exp_config.experiment_id,
        )

        try:
            exp_result = run_single_experiment(settings, exp_config)
            suite_result.completed += 1
        except Exception as e:
            logger.error(
                "Experiment %s failed: %s", exp_config.experiment_id, str(e),
            )
            exp_result = ExperimentResult(
                experiment_id=exp_config.experiment_id,
                algorithm=exp_config.algorithm,
                architecture="",
                model_label=exp_config.model.label,
                model_id=exp_config.model.model_id,
                seed=exp_config.seed,
                error=f"{type(e).__name__}: {str(e)}",
            )
            suite_result.failed += 1

        suite_result.experiments.append(exp_result)

        # Save checkpoint after each experiment
        write_json(str(checkpoint_path), suite_result.to_dict())
        logger.info(
            "Checkpoint saved (%d/%d complete).",
            idx + 1, len(suite.experiments),
        )

    # Save final results
    write_json(str(out_dir / "results.json"), suite_result.to_dict())

    return suite_result
