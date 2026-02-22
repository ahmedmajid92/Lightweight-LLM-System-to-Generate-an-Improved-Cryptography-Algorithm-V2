"""Safe dynamic compilation and registry injection for evolved components.

Compiles LLM-generated Python code in a sandboxed namespace, validates
it at the AST level to block dangerous operations, and injects verified
components into the live ComponentRegistry.

Research / education only. Do NOT use in production.
"""
from __future__ import annotations

import ast
import hashlib
import inspect
import random
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import sys
from pathlib import Path

_project_root = Path(__file__).parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from Components import (
    Component,
    ComponentRegistry,
    xor_bytes,
    rotate_left,
    rotate_right,
    bytes_to_words,
    words_to_bytes,
)
from cipherlab.config import Settings
from cipherlab.evolution.ast_analyzer import (
    DataFlowSignature,
    MismatchReport,
    analyze_component_code,
    detect_mismatches,
)
from cipherlab.evolution.component_mutator import (
    MutationRequest,
    MutationResult,
    mutate_component,
    mutate_with_retry,
    validate_mutation,
)


# ---------------------------------------------------------------------------
# Security exceptions
# ---------------------------------------------------------------------------

class SecurityError(Exception):
    """Raised when LLM-generated code contains unsafe constructs."""
    pass


class CompilationError(Exception):
    """Raised when LLM-generated code fails to compile or load."""
    pass


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class LoadedComponent:
    """A dynamically compiled and verified component ready for injection."""
    component_id: str
    kind: str
    source_code: str
    forward_func: Callable
    inverse_func: Optional[Callable]
    source_hash: str
    compatible_arch: Set[str]
    description: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "component_id": self.component_id,
            "kind": self.kind,
            "source_hash": self.source_hash,
            "compatible_arch": sorted(self.compatible_arch),
            "description": self.description,
        }


@dataclass
class EvolutionReport:
    """Report of all evolution attempts for a cipher spec."""
    timestamp: str = ""
    spec_name: str = ""
    mismatches_detected: List[MismatchReport] = field(default_factory=list)
    evolutions_attempted: int = 0
    evolutions_succeeded: int = 0
    evolved_components: List[LoadedComponent] = field(default_factory=list)
    failed_components: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.utcnow().isoformat()

    def all_resolved(self) -> bool:
        return len(self.failed_components) == 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "spec_name": self.spec_name,
            "mismatches_detected": [m.to_dict() for m in self.mismatches_detected],
            "evolutions_attempted": self.evolutions_attempted,
            "evolutions_succeeded": self.evolutions_succeeded,
            "evolved_components": [c.to_dict() for c in self.evolved_components],
            "failed_components": self.failed_components,
            "all_resolved": self.all_resolved(),
        }

    def summary(self) -> str:
        status = "ALL RESOLVED" if self.all_resolved() else "PARTIAL"
        return (
            f"[{status}] Evolution: {self.evolutions_succeeded}/{self.evolutions_attempted} "
            f"succeeded, {len(self.mismatches_detected)} mismatches detected"
        )


# ---------------------------------------------------------------------------
# AST safety validator
# ---------------------------------------------------------------------------

# Node types that are BLOCKED in LLM-generated code
_BLOCKED_NODE_TYPES = {
    ast.Import,
    ast.ImportFrom,
    ast.Global,
    ast.Nonlocal,
}

# Names that are BLOCKED from appearing in the code
_BLOCKED_NAMES = {
    "__builtins__", "__import__", "__loader__", "__spec__",
    "eval", "exec", "compile", "execfile",
    "open", "file", "input",
    "os", "sys", "subprocess", "shutil", "pathlib",
    "socket", "urllib", "requests", "http",
    "pickle", "shelve", "marshal",
    "ctypes", "cffi",
    "importlib", "runpy",
    "globals", "locals", "vars", "dir",
    "getattr", "setattr", "delattr",
    "type", "super", "__class__",
}

# Attributes that are BLOCKED
_BLOCKED_ATTRS = {
    "__builtins__", "__import__", "__loader__",
    "__subclasses__", "__bases__", "__mro__",
    "__code__", "__globals__", "__closure__",
}


def _validate_ast_safety(tree: ast.AST) -> List[str]:
    """Walk the AST and check for dangerous constructs.

    Returns a list of security violation descriptions. Empty = safe.
    """
    violations: List[str] = []

    for node in ast.walk(tree):
        # Check blocked node types
        for blocked_type in _BLOCKED_NODE_TYPES:
            if isinstance(node, blocked_type):
                violations.append(
                    f"Blocked construct: {blocked_type.__name__} at line {getattr(node, 'lineno', '?')}"
                )

        # Check Name nodes for blocked identifiers
        if isinstance(node, ast.Name) and node.id in _BLOCKED_NAMES:
            violations.append(
                f"Blocked name: '{node.id}' at line {getattr(node, 'lineno', '?')}"
            )

        # Check Attribute nodes for blocked attributes
        if isinstance(node, ast.Attribute) and node.attr in _BLOCKED_ATTRS:
            violations.append(
                f"Blocked attribute: '.{node.attr}' at line {getattr(node, 'lineno', '?')}"
            )

        # Check for Call to blocked builtins
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in _BLOCKED_NAMES:
                violations.append(
                    f"Blocked call: '{func.id}()' at line {getattr(node, 'lineno', '?')}"
                )

    return violations


# ---------------------------------------------------------------------------
# Safe compilation
# ---------------------------------------------------------------------------

def compile_component_code(
    source_code: str,
    expected_functions: List[str],
) -> Dict[str, Callable]:
    """Safely compile LLM-generated component code in a sandboxed namespace.

    Pipeline:
    1. Parse as AST
    2. Validate safety (no imports, no builtins access, etc.)
    3. Compile
    4. Execute in restricted namespace with only whitelisted utilities
    5. Extract expected functions

    Args:
        source_code: Python source code to compile.
        expected_functions: List of function names that must be defined.

    Returns:
        Dict mapping function name to callable.

    Raises:
        SecurityError: If code contains dangerous constructs.
        CompilationError: If code fails to compile or expected functions are missing.
    """
    # Step 1: Parse
    try:
        tree = ast.parse(source_code, mode="exec")
    except SyntaxError as e:
        raise CompilationError(f"Syntax error in generated code: {e}")

    # Step 2: AST safety validation
    violations = _validate_ast_safety(tree)
    if violations:
        raise SecurityError(
            "Security violations in generated code:\n" +
            "\n".join(f"  - {v}" for v in violations)
        )

    # Step 3: Compile
    try:
        code_obj = compile(tree, filename="<llm_component>", mode="exec")
    except Exception as e:
        raise CompilationError(f"Compilation failed: {e}")

    # Step 4: Execute in sandboxed namespace
    safe_globals: Dict[str, Any] = {
        "__builtins__": {},  # Empty builtins — no open(), exec(), etc.
        # Python built-in types (safe)
        "bytes": bytes,
        "bytearray": bytearray,
        "int": int,
        "float": float,
        "bool": bool,
        "str": str,
        "len": len,
        "range": range,
        "enumerate": enumerate,
        "zip": zip,
        "list": list,
        "tuple": tuple,
        "dict": dict,
        "set": set,
        "abs": abs,
        "min": min,
        "max": max,
        "sum": sum,
        "sorted": sorted,
        "reversed": reversed,
        "any": any,
        "all": all,
        "hex": hex,
        "bin": bin,
        "chr": chr,
        "ord": ord,
        "isinstance": isinstance,
        "ValueError": ValueError,
        "IndexError": IndexError,
        "TypeError": TypeError,
        # Crypto utilities from Components.py
        "xor_bytes": xor_bytes,
        "rotate_left": rotate_left,
        "rotate_right": rotate_right,
        "bytes_to_words": bytes_to_words,
        "words_to_bytes": words_to_bytes,
    }

    try:
        exec(code_obj, safe_globals)
    except Exception as e:
        raise CompilationError(f"Execution in sandbox failed: {e}")

    # Step 5: Extract expected functions
    result: Dict[str, Callable] = {}
    missing: List[str] = []
    for func_name in expected_functions:
        if func_name in safe_globals and callable(safe_globals[func_name]):
            result[func_name] = safe_globals[func_name]
        else:
            missing.append(func_name)

    if missing:
        available = [k for k in safe_globals if callable(safe_globals.get(k)) and not k.startswith("_")]
        raise CompilationError(
            f"Expected functions not found: {missing}. "
            f"Available functions in compiled scope: "
            f"{[k for k in available if k not in dir(__builtins__)]}"
        )

    return result


# ---------------------------------------------------------------------------
# Registry injection
# ---------------------------------------------------------------------------

def inject_component(
    registry: ComponentRegistry,
    loaded: LoadedComponent,
) -> None:
    """Inject a dynamically compiled component into the registry.

    Uses the existing ComponentRegistry.register() method which
    overwrites entries with the same component_id.

    Args:
        registry: The active component registry.
        loaded: The compiled and verified component to inject.
    """
    new_comp = Component(
        component_id=loaded.component_id,
        kind=loaded.kind,
        description=loaded.description,
        compatible_arch=loaded.compatible_arch,
        forward=loaded.forward_func,
        inverse=loaded.inverse_func,
    )
    registry.register(new_comp)


# ---------------------------------------------------------------------------
# Full pipeline orchestrator
# ---------------------------------------------------------------------------

def evolve_component(
    settings: Settings,
    spec,
    mismatch: MismatchReport,
    registry: ComponentRegistry,
    max_retries: int = 2,
) -> Optional[LoadedComponent]:
    """Detect → Mutate → Compile → Validate → Inject one component.

    Args:
        settings: Application settings with API keys.
        spec: CipherSpec that triggered the mismatch.
        mismatch: The detected I/O mismatch to resolve.
        registry: Active component registry.
        max_retries: Maximum LLM retry attempts on validation failure.

    Returns:
        LoadedComponent if successful, None if all retries failed.
    """
    comp_id = mismatch.component_id
    role = mismatch.role

    # Get original component
    if not registry.exists(comp_id):
        return None

    comp = registry.get(comp_id)

    # Extract source
    try:
        original_source = inspect.getsource(comp.forward)
    except (OSError, TypeError):
        return None

    # Get AST signature
    sig = analyze_component_code(original_source)

    # Get cryptographic profile (for S-boxes)
    preserve_ddt = None
    preserve_lat = None
    preserve_bij = comp.inverse is not None

    if comp.kind == "SBOX" and role in ("sbox", "f_sbox"):
        try:
            from cipherlab.evaluation.sbox_analysis import analyze_sbox
            sbox_result = analyze_sbox(comp_id, registry)
            preserve_ddt = sbox_result.ddt_max
            preserve_lat = sbox_result.lat_max_abs
            preserve_bij = sbox_result.is_bijective
        except Exception:
            pass

    # Derive target requirement
    from .ast_analyzer import derive_requirements
    reqs = derive_requirements(spec, registry)
    target_req = None
    for r in reqs:
        if r.role == role:
            target_req = r
            break

    if target_req is None:
        return None

    # Build mutation request
    mutation_req = MutationRequest(
        original_component_id=comp_id,
        original_source=original_source,
        original_signature=sig,
        target_requirement=target_req,
        mismatch=mismatch,
        preserve_ddt=preserve_ddt,
        preserve_lat=preserve_lat,
        preserve_bijectivity=preserve_bij,
    )

    # Attempt mutation with retries
    validation_errors: List[str] = []

    for attempt in range(max_retries + 1):
        # Get mutation result
        if attempt == 0:
            mut_result = mutate_component(settings, mutation_req)
        else:
            mut_result = mutate_with_retry(
                settings, mutation_req, validation_errors, max_retries=1,
            )

        if not mut_result.success:
            validation_errors.append(f"Attempt {attempt + 1}: LLM returned empty code")
            continue

        # Compile in sandbox
        expected_funcs = [mut_result.new_forward_name]
        if mut_result.new_inverse_name:
            expected_funcs.append(mut_result.new_inverse_name)

        try:
            compiled = compile_component_code(
                mut_result.new_source_code, expected_funcs
            )
        except SecurityError as e:
            validation_errors.append(f"Attempt {attempt + 1}: Security violation: {e}")
            continue
        except CompilationError as e:
            validation_errors.append(f"Attempt {attempt + 1}: Compilation failed: {e}")
            continue

        # Validate cryptographic properties
        passed, errors = validate_mutation(mut_result, mutation_req, compiled)
        mut_result.validation_passed = passed
        mut_result.validation_errors = errors

        if passed:
            # Build LoadedComponent
            source_hash = hashlib.sha256(
                mut_result.new_source_code.encode()
            ).hexdigest()[:16]

            loaded = LoadedComponent(
                component_id=mut_result.new_component_id,
                kind=comp.kind,
                source_code=mut_result.new_source_code,
                forward_func=compiled[mut_result.new_forward_name],
                inverse_func=(
                    compiled.get(mut_result.new_inverse_name)
                    if mut_result.new_inverse_name else None
                ),
                source_hash=source_hash,
                compatible_arch=set(comp.compatible_arch),
                description=(
                    f"Evolved from {comp_id}: adapted to "
                    f"{target_req.expected_input_bytes}B input, "
                    f"{target_req.expected_word_size_bits}b words"
                ),
            )

            # Inject into registry
            inject_component(registry, loaded)

            return loaded
        else:
            validation_errors.extend(errors)

    return None


def evolve_all_mismatches(
    settings: Settings,
    spec,
    registry: ComponentRegistry,
    max_retries: int = 2,
    blocking_only: bool = True,
) -> EvolutionReport:
    """Detect all mismatches and attempt to evolve each one.

    Args:
        settings: Application settings.
        spec: CipherSpec to check.
        registry: Active component registry.
        max_retries: Max LLM retries per component.
        blocking_only: If True, only evolve "blocking" mismatches.

    Returns:
        EvolutionReport with results of all evolution attempts.
    """
    mismatches = detect_mismatches(spec, registry)

    if blocking_only:
        mismatches = [m for m in mismatches if m.severity == "blocking"]

    report = EvolutionReport(
        spec_name=spec.name,
        mismatches_detected=mismatches,
    )

    for mm in mismatches:
        report.evolutions_attempted += 1

        loaded = evolve_component(
            settings, spec, mm, registry, max_retries=max_retries,
        )

        if loaded is not None:
            report.evolutions_succeeded += 1
            report.evolved_components.append(loaded)

            # Update the spec's component reference to use the new ID
            if mm.role in spec.components:
                spec.components[mm.role] = loaded.component_id
        else:
            report.failed_components.append(mm.component_id)

    return report
