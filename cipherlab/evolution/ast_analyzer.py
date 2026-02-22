"""AST-based dependency mapping and I/O mismatch detection.

Parses cipher component source code as Abstract Syntax Trees to extract
data flow signatures (input/output sizes, word sizes, bit operations)
and compares them against CipherConfiguration requirements to detect
mismatches before execution.

Research / education only. Do NOT use in production.
"""
from __future__ import annotations

import ast
import inspect
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Set, Tuple

import sys
from pathlib import Path

_project_root = Path(__file__).parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from Components import ComponentRegistry


# ---------------------------------------------------------------------------
# Known bitmask → word-size mappings
# ---------------------------------------------------------------------------

_MASK_TO_BITS: Dict[int, int] = {
    0xFF: 8,
    0xFFFF: 16,
    0xFFFFFFFF: 32,
    0xFFFFFFFFFFFFFFFF: 64,
    0x0F: 4,
}

_STRUCT_FMT_TO_BYTES: Dict[str, int] = {
    "B": 1, "b": 1,
    "H": 2, "h": 2,
    "I": 4, "i": 4,
    "L": 4, "l": 4,
    "Q": 8, "q": 8,
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class DataFlowSignature:
    """Extracted I/O signature of a cipher component function."""
    function_name: str
    input_byte_size: Optional[int] = None
    output_byte_size: Optional[int] = None
    word_size_bits: Optional[int] = None
    num_words: Optional[int] = None
    uses_struct: bool = False
    struct_format: Optional[str] = None
    bit_operations: List[str] = field(default_factory=list)
    hardcoded_constants: Dict[str, int] = field(default_factory=dict)
    length_checks: List[int] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ComponentRequirement:
    """What a cipher architecture requires from a component slot."""
    role: str
    expected_input_bytes: int
    expected_output_bytes: int
    expected_word_size_bits: int
    architecture: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MismatchReport:
    """Describes an I/O mismatch between a component and its requirement."""
    component_id: str
    role: str
    mismatch_type: str       # "input_size" | "output_size" | "word_size" | "length_check"
    expected: Any
    actual: Any
    severity: str            # "blocking" | "degraded"
    description: str
    suggested_fix: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def summary(self) -> str:
        return f"[{self.severity.upper()}] {self.component_id}: {self.description}"


# ---------------------------------------------------------------------------
# AST visitor for data flow extraction
# ---------------------------------------------------------------------------

class _DataFlowVisitor(ast.NodeVisitor):
    """Walk an AST to extract data flow characteristics."""

    def __init__(self):
        self.length_checks: List[int] = []
        self.word_size_bytes: Optional[int] = None
        self.struct_format: Optional[str] = None
        self.uses_struct: bool = False
        self.bit_ops: Set[str] = set()
        self.constants: Dict[str, int] = {}
        self.function_name: str = ""

    def visit_FunctionDef(self, node: ast.FunctionDef):
        if not self.function_name:
            self.function_name = node.name
        self.generic_visit(node)

    def visit_Compare(self, node: ast.Compare):
        """Detect length checks like `if len(data) != 8` or `len(data) == 16`."""
        left = node.left
        if isinstance(left, ast.Call) and isinstance(left.func, ast.Name):
            if left.func.id == "len":
                for comparator in node.comparators:
                    if isinstance(comparator, ast.Constant) and isinstance(comparator.value, int):
                        self.length_checks.append(comparator.value)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        """Detect bytes_to_words(data, N) and struct.unpack calls."""
        func = node.func
        func_name = ""
        if isinstance(func, ast.Name):
            func_name = func.id
        elif isinstance(func, ast.Attribute):
            func_name = func.attr

        # bytes_to_words(data, word_size_bytes, ...)
        if func_name == "bytes_to_words" and len(node.args) >= 2:
            arg = node.args[1]
            if isinstance(arg, ast.Constant) and isinstance(arg.value, int):
                self.word_size_bytes = arg.value
                self.uses_struct = True

        # words_to_bytes(words, word_size_bytes, ...)
        if func_name == "words_to_bytes" and len(node.args) >= 2:
            arg = node.args[1]
            if isinstance(arg, ast.Constant) and isinstance(arg.value, int):
                if self.word_size_bytes is None:
                    self.word_size_bytes = arg.value
                self.uses_struct = True

        # struct.unpack(fmt, ...)
        if func_name == "unpack" and node.args:
            fmt_arg = node.args[0]
            if isinstance(fmt_arg, ast.Constant) and isinstance(fmt_arg.value, str):
                self.struct_format = fmt_arg.value
                self.uses_struct = True
                for ch in fmt_arg.value:
                    if ch in _STRUCT_FMT_TO_BYTES:
                        self.word_size_bytes = _STRUCT_FMT_TO_BYTES[ch]
                        break

        # int.from_bytes(data, ...) — indicates full-block integer conversion
        if func_name == "from_bytes":
            self.uses_struct = True

        # rotate_left / rotate_right calls
        if func_name in ("rotate_left", "rotate_right"):
            self.bit_ops.add("rotate")
            # Third arg is word_size in bits
            if len(node.args) >= 3:
                arg = node.args[2]
                if isinstance(arg, ast.Constant) and isinstance(arg.value, int):
                    ws_bits = arg.value
                    if self.word_size_bytes is None:
                        self.word_size_bytes = ws_bits // 8

        self.generic_visit(node)

    def visit_BinOp(self, node: ast.BinOp):
        """Detect bit operations and extract mask constants."""
        op = node.op
        if isinstance(op, ast.LShift):
            self.bit_ops.add("shift_left")
        elif isinstance(op, ast.RShift):
            self.bit_ops.add("shift_right")
        elif isinstance(op, ast.BitAnd):
            self.bit_ops.add("and")
            # Check for known mask constants
            for operand in (node.left, node.right):
                if isinstance(operand, ast.Constant) and isinstance(operand.value, int):
                    val = operand.value
                    if val in _MASK_TO_BITS:
                        bits = _MASK_TO_BITS[val]
                        self.constants[f"mask_0x{val:X}"] = val
                        if self.word_size_bytes is None or bits // 8 > 0:
                            # Only update if we detect a larger mask
                            candidate = bits // 8 if bits >= 8 else 1
                            if self.word_size_bytes is None:
                                self.word_size_bytes = candidate
        elif isinstance(op, ast.BitXor):
            self.bit_ops.add("xor")
        elif isinstance(op, ast.BitOr):
            self.bit_ops.add("or")
        elif isinstance(op, ast.Add):
            self.bit_ops.add("add")
        elif isinstance(op, ast.Sub):
            self.bit_ops.add("sub")
        elif isinstance(op, ast.Mod):
            self.bit_ops.add("mod")

        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant):
        """Record notable integer constants."""
        if isinstance(node.value, int) and node.value > 0:
            val = node.value
            if val in _MASK_TO_BITS:
                self.constants[f"mask_0x{val:X}"] = val

    def build_signature(self) -> DataFlowSignature:
        word_bits = (self.word_size_bytes * 8) if self.word_size_bytes else None
        # Input/output size inferred from length checks
        input_size = self.length_checks[0] if self.length_checks else None
        output_size = input_size  # Components are bytes->bytes, same size

        return DataFlowSignature(
            function_name=self.function_name,
            input_byte_size=input_size,
            output_byte_size=output_size,
            word_size_bits=word_bits,
            num_words=None,
            uses_struct=self.uses_struct,
            struct_format=self.struct_format,
            bit_operations=sorted(self.bit_ops),
            hardcoded_constants=self.constants,
            length_checks=sorted(set(self.length_checks)),
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyze_component_code(source_code: str) -> DataFlowSignature:
    """Parse Python source code and extract its data flow signature.

    Args:
        source_code: Python function source code.

    Returns:
        DataFlowSignature with extracted I/O characteristics.
    """
    try:
        tree = ast.parse(source_code)
    except SyntaxError:
        return DataFlowSignature(function_name="<parse_error>")

    visitor = _DataFlowVisitor()
    visitor.visit(tree)
    return visitor.build_signature()


def derive_requirements(
    spec,
    registry: Optional[ComponentRegistry] = None,
) -> List[ComponentRequirement]:
    """Derive I/O requirements for each component slot in a cipher spec.

    Args:
        spec: CipherSpec (from AlgorithmsBlock or cipherlab.cipher.spec).
        registry: Optional component registry (unused currently, reserved).

    Returns:
        List of ComponentRequirement, one per component slot.
    """
    block_bytes = spec.block_size_bits // 8
    key_bytes = spec.key_size_bits // 8
    half_block = block_bytes // 2
    word_size = getattr(spec, "word_size", 32)
    arch = spec.architecture.upper()

    reqs: List[ComponentRequirement] = []

    if arch == "SPN":
        for role in ("sbox", "perm", "linear"):
            if role in spec.components:
                reqs.append(ComponentRequirement(
                    role=role,
                    expected_input_bytes=block_bytes,
                    expected_output_bytes=block_bytes,
                    expected_word_size_bits=word_size,
                    architecture=arch,
                ))
    elif arch == "FEISTEL":
        for role in ("f_sbox", "f_perm"):
            if role in spec.components:
                reqs.append(ComponentRequirement(
                    role=role,
                    expected_input_bytes=half_block,
                    expected_output_bytes=half_block,
                    expected_word_size_bits=word_size,
                    architecture=arch,
                ))
    elif arch == "ARX":
        for role in ("arx_add", "arx_rotate"):
            if role in spec.components:
                reqs.append(ComponentRequirement(
                    role=role,
                    expected_input_bytes=block_bytes,
                    expected_output_bytes=block_bytes,
                    expected_word_size_bits=word_size,
                    architecture=arch,
                ))

    # Key schedule requirement (all architectures)
    if "key_schedule" in spec.components:
        out_bytes = half_block if arch == "FEISTEL" else block_bytes
        reqs.append(ComponentRequirement(
            role="key_schedule",
            expected_input_bytes=key_bytes,
            expected_output_bytes=out_bytes,
            expected_word_size_bits=word_size,
            architecture=arch,
        ))

    return reqs


def detect_mismatches(
    spec,
    registry: Optional[ComponentRegistry] = None,
) -> List[MismatchReport]:
    """Detect I/O mismatches between spec components and architecture requirements.

    Args:
        spec: CipherSpec with components dict.
        registry: Component registry to look up component source code.

    Returns:
        List of MismatchReport sorted by severity (blocking first).
    """
    reg = registry or ComponentRegistry()
    requirements = derive_requirements(spec, reg)
    reports: List[MismatchReport] = []

    # Build a map of role -> requirement
    req_by_role: Dict[str, ComponentRequirement] = {r.role: r for r in requirements}

    for role, comp_id in spec.components.items():
        if role == "key_schedule":
            continue  # Key schedules have different signature, skip AST analysis

        if not reg.exists(comp_id):
            reports.append(MismatchReport(
                component_id=comp_id,
                role=role,
                mismatch_type="missing",
                expected="registered component",
                actual="not found",
                severity="blocking",
                description=f"Component '{comp_id}' not found in registry",
                suggested_fix=f"Register '{comp_id}' or use a different component",
            ))
            continue

        req = req_by_role.get(role)
        if req is None:
            continue

        comp = reg.get(comp_id)

        # Extract source code
        try:
            source = inspect.getsource(comp.forward)
        except (OSError, TypeError):
            continue  # Can't analyze without source

        sig = analyze_component_code(source)

        # Check 1: Length checks vs expected input size
        if sig.length_checks:
            for lc in sig.length_checks:
                if lc != req.expected_input_bytes:
                    reports.append(MismatchReport(
                        component_id=comp_id,
                        role=role,
                        mismatch_type="length_check",
                        expected=req.expected_input_bytes,
                        actual=lc,
                        severity="blocking",
                        description=(
                            f"Component enforces len(data)=={lc} but architecture "
                            f"requires {req.expected_input_bytes} bytes "
                            f"(block={spec.block_size_bits}b)"
                        ),
                        suggested_fix=(
                            f"Adapt '{comp_id}' to accept {req.expected_input_bytes}-byte input, "
                            f"or use a component compatible with {spec.block_size_bits}-bit blocks"
                        ),
                    ))

        # Check 2: Word size mismatch
        if sig.word_size_bits is not None and sig.word_size_bits != req.expected_word_size_bits:
            reports.append(MismatchReport(
                component_id=comp_id,
                role=role,
                mismatch_type="word_size",
                expected=req.expected_word_size_bits,
                actual=sig.word_size_bits,
                severity="degraded",
                description=(
                    f"Component uses {sig.word_size_bits}-bit words but architecture "
                    f"expects {req.expected_word_size_bits}-bit words"
                ),
                suggested_fix=(
                    f"Adapt '{comp_id}' to use {req.expected_word_size_bits}-bit word operations"
                ),
            ))

        # Check 3: Hardcoded mask mismatch
        expected_mask = (1 << req.expected_word_size_bits) - 1
        for const_name, const_val in sig.hardcoded_constants.items():
            if const_val in _MASK_TO_BITS:
                detected_bits = _MASK_TO_BITS[const_val]
                if detected_bits != req.expected_word_size_bits and detected_bits >= 8:
                    # Only flag if we haven't already flagged word_size
                    already_flagged = any(
                        r.component_id == comp_id and r.mismatch_type == "word_size"
                        for r in reports
                    )
                    if not already_flagged:
                        reports.append(MismatchReport(
                            component_id=comp_id,
                            role=role,
                            mismatch_type="word_size",
                            expected=req.expected_word_size_bits,
                            actual=detected_bits,
                            severity="degraded",
                            description=(
                                f"Component uses hardcoded {detected_bits}-bit mask "
                                f"(0x{const_val:X}) but architecture expects "
                                f"{req.expected_word_size_bits}-bit words"
                            ),
                            suggested_fix=(
                                f"Replace mask 0x{const_val:X} with "
                                f"0x{expected_mask:X} ({req.expected_word_size_bits}-bit)"
                            ),
                        ))

    # Sort: blocking first, then degraded
    severity_order = {"blocking": 0, "degraded": 1}
    reports.sort(key=lambda r: severity_order.get(r.severity, 2))

    return reports


# ---------------------------------------------------------------------------
# LLM-generated code analysis
# ---------------------------------------------------------------------------

_ROLE_PATTERNS: Dict[str, List[str]] = {
    "sbox": ["sbox", "substitut", "s_box", "sub_bytes"],
    "perm": ["perm", "shiftrow", "bit_perm", "shuffle"],
    "linear": ["mix", "linear", "diffus", "mds"],
    "arx_add": ["add_mod", "mod_add", "modular_add"],
    "arx_rotate": ["rotat", "rol", "ror", "circular"],
    "key_schedule": ["key_sched", "kdf", "expand_key", "key_expansion"],
}


def analyze_llm_cipher_code(code_string: str) -> Dict[str, DataFlowSignature]:
    """Analyze LLM-generated cipher code and identify component functions.

    Args:
        code_string: Full Python module source code from LLM.

    Returns:
        Dict mapping detected role to DataFlowSignature.
    """
    try:
        tree = ast.parse(code_string)
    except SyntaxError:
        return {}

    results: Dict[str, DataFlowSignature] = {}

    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue

        func_name = node.name.lower()

        # Detect role from function name
        detected_role = None
        for role, patterns in _ROLE_PATTERNS.items():
            if any(p in func_name for p in patterns):
                detected_role = role
                break

        if detected_role is None:
            continue

        # Extract the function source from the full code
        try:
            func_source = ast.get_source_segment(code_string, node)
            if func_source:
                sig = analyze_component_code(func_source)
                sig.function_name = node.name
                results[detected_role] = sig
        except Exception:
            continue

    return results
