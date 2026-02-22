"""Deterministic feedback synthesis: traceback parser + DeepSeek-R1 integration.

Parses evaluation results into structured diagnostics, builds targeted prompts,
and autonomously feeds them to DeepSeek-R1 (via OpenRouter) or OpenAI
for improvement suggestions.

Research / education only. Do NOT use in production.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

from cipherlab.config import Settings
from cipherlab.cipher.spec import CipherSpec, ImprovementPatch
from cipherlab.llm.openai_provider import OpenAIProvider
from .report import EvaluationReport
from .roundtrip import RoundtripResult
from .avalanche import SACResult
from .sbox_analysis import SBoxAnalysisResult


# ---------------------------------------------------------------------------
# Diagnostic dataclass
# ---------------------------------------------------------------------------

@dataclass
class EvaluationDiagnostic:
    """Structured diagnostic extracted from evaluation failures."""
    severity: str           # "critical" | "warning" | "info"
    category: str           # "roundtrip_failure" | "sac_weak_bits" | "sbox_differential" | "low_avalanche"
    algorithm: str          # Algorithm or component name
    description: str        # Human-readable one-liner
    data: Dict[str, Any] = field(default_factory=dict)
    suggested_focus: str = ""  # Which component/area to investigate

    def to_prompt_block(self) -> str:
        """Format as structured text block for LLM prompt injection."""
        lines = [
            f"[{self.severity.upper()}] {self.category}: {self.description}",
            f"  Algorithm: {self.algorithm}",
            f"  Suggested focus: {self.suggested_focus}",
        ]
        if self.data:
            for k, v in self.data.items():
                lines.append(f"  {k}: {v}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Feedback cycle result
# ---------------------------------------------------------------------------

@dataclass
class FeedbackCycleResult:
    """Result of one feedback cycle: diagnostics → prompt → LLM → patch."""
    diagnostics: List[EvaluationDiagnostic] = field(default_factory=list)
    prompt_system: str = ""
    prompt_user: str = ""
    patch: Optional[ImprovementPatch] = None
    raw_response: Any = None
    model_used: str = ""
    reasoning_trace: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "diagnostics": [d.to_dict() for d in self.diagnostics],
            "patch": self.patch.model_dump() if self.patch else None,
            "model_used": self.model_used,
            "has_reasoning_trace": self.reasoning_trace is not None,
        }


# ---------------------------------------------------------------------------
# Diagnostic parser
# ---------------------------------------------------------------------------

def parse_evaluation_results(report: EvaluationReport) -> List[EvaluationDiagnostic]:
    """Parse an EvaluationReport into structured diagnostics.

    Extraction rules:
    1. Roundtrip failures → severity="critical"
    2. SAC weak bits → severity="warning"
    3. S-box high DDT/LAT → severity="warning"
    4. Low overall avalanche → severity="warning"
    """
    diagnostics: List[EvaluationDiagnostic] = []

    # 1. Roundtrip failures (CRITICAL)
    for rt in report.roundtrip_results:
        if not rt.is_perfect:
            failure_samples = rt.failures[:3]
            sample_data = [
                {
                    "vector": f.vector_index,
                    "pt": f.plaintext_hex[:32] + "..." if len(f.plaintext_hex) > 32 else f.plaintext_hex,
                    "error": f.error,
                }
                for f in failure_samples
            ]
            diagnostics.append(EvaluationDiagnostic(
                severity="critical",
                category="roundtrip_failure",
                algorithm=rt.algorithm_name,
                description=(
                    f"Roundtrip P=D(E(P,K),K) failed for {rt.failed}/{rt.total_vectors} vectors"
                ),
                data={
                    "failed_count": rt.failed,
                    "total_vectors": rt.total_vectors,
                    "architecture": rt.architecture,
                    "sample_failures": sample_data,
                },
                suggested_focus=f"{rt.architecture} encrypt/decrypt implementation",
            ))

    # 2. SAC weak bits (WARNING)
    for sac in report.sac_results:
        if not sac.passes_sac:
            # Find specific weak bit positions
            weak_bits = [
                i for i, p in enumerate(sac.per_input_bit_mean) if p < 0.35
            ]
            diagnostics.append(EvaluationDiagnostic(
                severity="warning",
                category="sac_weak_bits",
                algorithm=sac.algorithm_name,
                description=(
                    f"SAC({sac.input_type}) failed: deviation={sac.sac_deviation:.4f}, "
                    f"min_bit={sac.min_bit_prob:.4f}, {len(weak_bits)} weak bit positions"
                ),
                data={
                    "input_type": sac.input_type,
                    "global_mean": sac.global_mean,
                    "sac_deviation": sac.sac_deviation,
                    "min_bit_prob": sac.min_bit_prob,
                    "max_bit_prob": sac.max_bit_prob,
                    "weak_bit_positions": weak_bits[:20],  # Limit for prompt size
                    "num_weak_bits": len(weak_bits),
                },
                suggested_focus=(
                    "permutation or linear diffusion layer"
                    if sac.input_type == "plaintext"
                    else "key schedule"
                ),
            ))

    # 3. S-box differential/linear weaknesses (WARNING)
    for sb in report.sbox_results:
        issues = []
        if sb.differential_uniformity == "poor":
            issues.append(f"DDT_max={sb.ddt_max} (poor)")
        if sb.linearity == "poor":
            issues.append(f"LAT_max={sb.lat_max_abs} (poor)")
        if not sb.is_bijective:
            issues.append("NOT bijective")

        if issues:
            diagnostics.append(EvaluationDiagnostic(
                severity="warning",
                category="sbox_differential",
                algorithm=sb.component_id,
                description=f"S-box weakness: {'; '.join(issues)}",
                data={
                    "sbox_size": sb.sbox_size,
                    "ddt_max": sb.ddt_max,
                    "lat_max_abs": sb.lat_max_abs,
                    "is_bijective": sb.is_bijective,
                    "differential_uniformity": sb.differential_uniformity,
                    "linearity": sb.linearity,
                },
                suggested_focus="sbox replacement",
            ))

    # 4. Low overall avalanche from SAC global_mean (WARNING)
    for sac in report.sac_results:
        if sac.global_mean < 0.40:
            diagnostics.append(EvaluationDiagnostic(
                severity="warning",
                category="low_avalanche",
                algorithm=sac.algorithm_name,
                description=(
                    f"Low avalanche ({sac.input_type}): mean={sac.global_mean:.4f} "
                    f"(target ~0.50)"
                ),
                data={
                    "input_type": sac.input_type,
                    "global_mean": sac.global_mean,
                },
                suggested_focus="rounds or key_schedule",
            ))

    # Sort: critical first, then warning, then info
    severity_order = {"critical": 0, "warning": 1, "info": 2}
    diagnostics.sort(key=lambda d: severity_order.get(d.severity, 3))

    return diagnostics


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

FEEDBACK_SYSTEM = """You are a cryptographic engineering assistant specializing in lightweight block ciphers for IoT and resource-constrained environments.

You are analyzing automated evaluation results for a cipher design. Your task is to diagnose issues and propose fixes.

Rules:
- CRITICAL failures (roundtrip P≠D(E(P,K),K)) must be resolved first — these indicate fundamental correctness bugs.
- Then address WARNING issues (SAC deviations, S-box weaknesses, low avalanche).
- Do NOT claim security. Phrase all suggestions as hypotheses for further testing.
- Only suggest components from the available component list.
- Your response must be a valid ImprovementPatch JSON matching the schema exactly.
- Include specific measurement values in your rationale to justify each change.
"""


def build_feedback_prompt(
    spec: CipherSpec,
    diagnostics: List[EvaluationDiagnostic],
    rag_context: str,
    available_components: List[str],
) -> Tuple[str, str]:
    """Build (system_prompt, user_prompt) for DeepSeek-R1 feedback.

    Args:
        spec: Current cipher specification.
        diagnostics: Parsed evaluation diagnostics.
        rag_context: Retrieved KB context string.
        available_components: List of available component IDs.

    Returns:
        Tuple of (system_prompt, user_prompt).
    """
    system = FEEDBACK_SYSTEM

    # Build user prompt
    sections = []

    # Current cipher spec
    sections.append("## Current Cipher Configuration\n```json\n" + spec.model_dump_json(indent=2) + "\n```")

    # Diagnostics
    if diagnostics:
        sections.append("## Evaluation Diagnostics\n")
        for d in diagnostics:
            sections.append(d.to_prompt_block())
            sections.append("")
    else:
        sections.append("## Evaluation Diagnostics\nNo issues detected. The cipher passes all checks.")

    # RAG context
    if rag_context:
        sections.append("## RAG Context (Lightweight Cipher Literature)\n" + rag_context)

    # Available components
    sections.append("## Available Components\n" + "\n".join(f"- {c}" for c in available_components))

    # Task instruction
    sections.append(
        "## Task\n"
        "Analyze the diagnostics above. Provide an ImprovementPatch JSON that:\n"
        "1. Addresses all CRITICAL issues first (roundtrip correctness)\n"
        "2. Then optimizes for SAC/avalanche warnings\n"
        "3. Explains rationale citing specific measurements from the diagnostics\n"
        "4. Only suggests components from the available list above\n\n"
        "If no issues are found, suggest optimizations to bring SAC deviation closer to 0.0."
    )

    user = "\n\n".join(sections)

    return system, user


# ---------------------------------------------------------------------------
# Autonomous feedback loop
# ---------------------------------------------------------------------------

def run_feedback_cycle(
    settings: Settings,
    spec: CipherSpec,
    report: EvaluationReport,
    rag_context: str = "",
    available_components: Optional[List[str]] = None,
) -> FeedbackCycleResult:
    """Run one feedback cycle: parse → prompt → call DeepSeek-R1 → return patch.

    Uses OpenRouter (DeepSeek-R1 reasoning model) when available,
    falls back to OpenAI quality model.

    Args:
        settings: Application settings with API keys.
        spec: Current cipher specification.
        report: Complete evaluation report.
        rag_context: Optional RAG context string.
        available_components: Optional list of component IDs. Defaults to
            a standard set if not provided.

    Returns:
        FeedbackCycleResult with diagnostics, prompts, and improvement patch.
    """
    if available_components is None:
        available_components = [
            "ks.sha256_kdf", "ks.des_style", "ks.blowfish_style",
            "sbox.aes", "sbox.present", "sbox.gift", "sbox.identity",
            "sbox.des", "sbox.blowfish", "sbox.serpent",
            "sbox.tea_f", "sbox.xtea_f", "sbox.simon_f", "sbox.hight_f",
            "perm.aes_shiftrows", "perm.present", "perm.gift",
            "perm.identity", "perm.des_ip", "perm.serpent",
            "linear.aes_mixcolumns", "linear.identity", "linear.twofish_mds",
            "arx.add_mod32", "arx.rotate_left_3", "arx.rotate_left_5",
            "arx.mul_mod16",
        ]

    # Step 1: Parse diagnostics
    diagnostics = parse_evaluation_results(report)

    # Step 2: Build prompt
    system, user = build_feedback_prompt(spec, diagnostics, rag_context, available_components)

    # Step 3: Call LLM
    provider = OpenAIProvider(
        api_key=settings.openai_api_key,
        openrouter_api_key=settings.openrouter_api_key,
    )

    reasoning_trace = None

    if provider.openrouter_client and settings.openrouter_api_key:
        # Use DeepSeek-R1 via OpenRouter for chain-of-thought reasoning
        model_used = settings.openrouter_model_reasoning
        patch, raw = provider.generate_json_chat(
            model=model_used,
            system=system,
            user=user,
            schema=ImprovementPatch,
            temperature=0.2,
            max_tokens=4096,
            use_openrouter=True,
        )
        # Extract reasoning trace if available (DeepSeek-R1 may include it)
        if hasattr(raw, 'choices') and raw.choices:
            content = raw.choices[0].message.content or ""
            if "<think>" in content:
                start = content.index("<think>") + len("<think>")
                end = content.index("</think>") if "</think>" in content else len(content)
                reasoning_trace = content[start:end].strip()
    else:
        # Fall back to OpenAI Structured Outputs
        model_used = settings.openai_model_quality
        patch, raw = provider.generate_structured(
            model=model_used,
            system=system,
            user=user,
            schema=ImprovementPatch,
            temperature=0.2,
            max_output_tokens=1200,
        )

    return FeedbackCycleResult(
        diagnostics=diagnostics,
        prompt_system=system,
        prompt_user=user,
        patch=patch,
        raw_response=raw,
        model_used=model_used,
        reasoning_trace=reasoning_trace,
    )
