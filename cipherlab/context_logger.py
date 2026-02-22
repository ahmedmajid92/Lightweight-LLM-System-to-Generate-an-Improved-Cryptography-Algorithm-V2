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
            pt_av = self.metrics.get('pt_avalanche', self.metrics.get('plaintext_avalanche_mean', 'N/A'))
            key_av = self.metrics.get('key_avalanche', self.metrics.get('key_avalanche_mean', 'N/A'))
            score = self.metrics.get('overall_score', 'N/A')
            lines.append(f"Metrics: pt_avalanche={pt_av}, key_avalanche={key_av}, score={score}")
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
