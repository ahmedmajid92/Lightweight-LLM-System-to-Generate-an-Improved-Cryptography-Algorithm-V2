"""Context logger for capturing cipher configuration state and component signatures.

Provides structured context snapshots that can be injected into LLM prompts
to maintain awareness of the current cipher design during chat interactions.

Research / education only. Do NOT use in production.
"""
from __future__ import annotations

import ast
import inspect
import hashlib
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root for Components import
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from Components import Component, ComponentRegistry


@dataclass
class ComponentSignature:
    """AST-derived signature of a component function."""
    component_id: str
    kind: str
    function_name: str
    parameters: List[str]
    source_hash: str
    docstring: Optional[str] = None
    line_count: int = 0


@dataclass
class CipherContextSnapshot:
    """Complete snapshot of cipher state for LLM context injection."""
    timestamp: str
    cipher_name: str
    architecture: str
    configuration: Dict[str, Any]
    component_signatures: List[Dict[str, Any]]
    metrics: Optional[Dict[str, Any]] = None
    issues: List[str] = field(default_factory=list)

    def to_prompt_context(self) -> str:
        """Format snapshot as concise text for LLM system/user prompt."""
        cfg = self.configuration
        lines = [
            f"Current cipher: {self.cipher_name} ({self.architecture})",
            f"Block: {cfg['block_size_bits']}b, "
            f"Key: {cfg['key_size_bits']}b, "
            f"Rounds: {cfg['rounds']}, "
            f"Word: {cfg.get('word_size', 32)}b",
            "Components:",
        ]
        for sig in self.component_signatures:
            desc = sig.get('docstring', 'N/A')
            lines.append(f"  - {sig['component_id']} ({sig['kind']}): {desc}")
        if self.metrics:
            # Handle both flat and nested metric dicts
            pt_raw = self.metrics.get('plaintext_avalanche', {})
            key_raw = self.metrics.get('key_avalanche', {})
            scores = self.metrics.get('scores', {})
            pt_av = pt_raw.get('mean', 'N/A') if isinstance(pt_raw, dict) else pt_raw
            key_av = key_raw.get('mean', 'N/A') if isinstance(key_raw, dict) else key_raw
            score = scores.get('overall', 'N/A') if isinstance(scores, dict) else scores
            pt_str = f"{pt_av:.4f}" if isinstance(pt_av, (int, float)) else str(pt_av)
            key_str = f"{key_av:.4f}" if isinstance(key_av, (int, float)) else str(key_av)
            score_str = f"{score:.4f}" if isinstance(score, (int, float)) else str(score)
            lines.append(f"Metrics: pt_avalanche={pt_str}, key_avalanche={key_str}, overall_score={score_str}")
        if self.issues:
            lines.append("Issues: " + "; ".join(self.issues))
        return "\n".join(lines)


def extract_component_signature(comp: Component) -> ComponentSignature:
    """Extract AST signature from a component's forward function."""
    func = comp.forward
    source_code = ""
    try:
        source_code = inspect.getsource(func)
    except (OSError, TypeError):
        pass

    source_hash = hashlib.sha256(source_code.encode()).hexdigest()[:16]

    params: List[str] = []
    line_count = 0
    try:
        tree = ast.parse(source_code)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                params = [arg.arg for arg in node.args.args]
                line_count = len(source_code.splitlines())
                break
    except SyntaxError:
        pass

    docstring = None
    if func.__doc__:
        docstring = func.__doc__.strip().split("\n")[0]

    return ComponentSignature(
        component_id=comp.component_id,
        kind=comp.kind,
        function_name=func.__name__,
        parameters=params,
        source_hash=source_hash,
        docstring=docstring,
        line_count=line_count,
    )


def build_context_snapshot(
    spec,
    registry: ComponentRegistry,
    metrics: Optional[Dict] = None,
    issues: Optional[List[str]] = None,
) -> CipherContextSnapshot:
    """Build a complete context snapshot from current cipher state."""
    cfg = {
        "block_size_bits": spec.block_size_bits,
        "key_size_bits": spec.key_size_bits,
        "rounds": spec.rounds,
        "word_size": getattr(spec, "word_size", 32),
    }

    sigs = []
    for role, comp_id in spec.components.items():
        if registry.exists(comp_id):
            comp = registry.get(comp_id)
            sig = extract_component_signature(comp)
            sigs.append(asdict(sig))

    return CipherContextSnapshot(
        timestamp=datetime.utcnow().isoformat(),
        cipher_name=spec.name,
        architecture=spec.architecture,
        configuration=cfg,
        component_signatures=sigs,
        metrics=metrics,
        issues=issues or [],
    )


# ---------------------------------------------------------------------------
# Tiered context for design-review copilot (Option A)
# ---------------------------------------------------------------------------

def build_copilot_context(
    spec,
    registry: ComponentRegistry,
    metrics: Optional[Dict] = None,
    issues: Optional[List[str]] = None,
    iteration_history=None,
    diagnostics: Optional[List] = None,
    pending_patch: Optional[Dict] = None,
) -> str:
    """Build tiered context string for the design-review copilot.

    Tier 1 (always): working spec summary + iteration history summary (~500 tokens)
    Tier 2 (if available): active diagnostics + last patch details (~300 tokens)

    RAG KB snippets are added separately by the caller.

    Args:
        spec: Current working CipherSpec.
        registry: Component registry.
        metrics: Basic avalanche metrics dict (from evaluate_and_score).
        issues: Heuristic issue strings.
        iteration_history: IterationHistory instance (optional).
        diagnostics: List of EvaluationDiagnostic (optional).
        pending_patch: Dict with pending patch info (optional).

    Returns:
        Formatted context string for system prompt injection.
    """
    sections: List[str] = []

    # --- Tier 1: Always included ---

    # Working spec summary
    snapshot = build_context_snapshot(spec, registry, metrics, issues)
    sections.append("## Current Working Design\n" + snapshot.to_prompt_context())

    # Iteration history summary
    if iteration_history is not None:
        hist_text = iteration_history.to_context_summary(max_records=8)
        if hist_text and "No improvement" not in hist_text:
            sections.append("## Improvement History\n" + hist_text)

    # --- Tier 2: If available ---

    # Active diagnostics (up to 5)
    if diagnostics:
        diag_lines = []
        for d in diagnostics[:5]:
            diag_lines.append(d.to_prompt_block())
        sections.append("## Active Diagnostics\n" + "\n".join(diag_lines))

    # Last pending/staged patch
    if pending_patch:
        patch_obj = pending_patch.get("patch")
        if patch_obj:
            patch_lines = [f"Summary: {patch_obj.summary}"]
            if hasattr(patch_obj, "rationale") and patch_obj.rationale:
                for r in patch_obj.rationale[:3]:
                    patch_lines.append(f"- {r}")
            patch_lines.append(f"Model: {pending_patch.get('model_used', 'unknown')}")
            sections.append("## Pending Patch Under Review\n" + "\n".join(patch_lines))

    return "\n\n".join(sections)
