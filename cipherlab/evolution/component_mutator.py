"""Autonomous sub-agent for component mutation via DeepSeek-R1.

When an I/O mismatch is detected, this module constructs a targeted prompt
containing the original component's AST signature and the new I/O requirements,
then instructs DeepSeek-R1 to rewrite the component while preserving its
cryptographic properties (DDT/LAT profile, bijectivity).

Research / education only. Do NOT use in production.
"""
from __future__ import annotations

import inspect
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from cipherlab.config import Settings
from cipherlab.llm.openai_provider import OpenAIProvider
from .ast_analyzer import DataFlowSignature, ComponentRequirement, MismatchReport


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class MutationRequest:
    """Request to mutate a component to fit new I/O requirements."""
    original_component_id: str
    original_source: str
    original_signature: DataFlowSignature
    target_requirement: ComponentRequirement
    mismatch: MismatchReport
    preserve_ddt: Optional[int] = None
    preserve_lat: Optional[int] = None
    preserve_bijectivity: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_component_id": self.original_component_id,
            "target_role": self.target_requirement.role,
            "target_input_bytes": self.target_requirement.expected_input_bytes,
            "target_word_size_bits": self.target_requirement.expected_word_size_bits,
            "mismatch_type": self.mismatch.mismatch_type,
            "preserve_ddt": self.preserve_ddt,
            "preserve_lat": self.preserve_lat,
        }


@dataclass
class MutationResult:
    """Result of a component mutation attempt."""
    success: bool
    new_component_id: str = ""
    new_source_code: str = ""
    new_forward_name: str = ""
    new_inverse_name: Optional[str] = None
    reasoning_trace: Optional[str] = None
    validation_passed: bool = False
    validation_errors: List[str] = field(default_factory=list)
    model_used: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "new_component_id": self.new_component_id,
            "validation_passed": self.validation_passed,
            "validation_errors": self.validation_errors,
            "model_used": self.model_used,
            "has_reasoning_trace": self.reasoning_trace is not None,
        }


# ---------------------------------------------------------------------------
# Mutation prompt builder
# ---------------------------------------------------------------------------

MUTATION_SYSTEM = """You are a cryptographic component engineer specializing in lightweight block ciphers.
You must rewrite a cipher component function to work with different I/O dimensions
while preserving its core cryptographic properties.

RULES:
1. The rewritten functions must accept `data: bytes` and return `bytes`.
2. Input and output byte lengths must match the target requirement exactly.
3. If the original is a substitution box (S-box), preserve the same algebraic structure
   (differential uniformity, linearity profile) at the new dimension.
4. If the original is a bit permutation, create a mathematically valid permutation
   at the new bit width using the same mathematical pattern (e.g., P(i) = (k*i) mod (n-1)).
5. If the original uses word operations (bytes_to_words, rotate_left, etc.),
   adjust the word size parameter accordingly.
6. You MUST provide both a forward function and an inverse function.
7. Output ONLY valid Python code. No markdown fences. No explanations outside code comments.
8. You may use these utility functions (they are pre-loaded):
   - xor_bytes(a, b) -> bytes
   - rotate_left(x, r, w) -> int  (rotate integer x left by r bits in w-bit word)
   - rotate_right(x, r, w) -> int
   - bytes_to_words(data, word_size_bytes, byteorder) -> List[int]
   - words_to_bytes(words, word_size_bytes, byteorder) -> bytes
9. Include all lookup tables and constants inline in the code.
"""


def build_mutation_prompt(request: MutationRequest) -> Tuple[str, str]:
    """Build (system_prompt, user_prompt) for DeepSeek-R1 component mutation.

    Args:
        request: MutationRequest with original component info and target requirements.

    Returns:
        Tuple of (system_prompt, user_prompt).
    """
    req = request.target_requirement
    sig = request.original_signature
    mm = request.mismatch

    # Derive new function names
    base_name = request.original_component_id.replace(".", "_")
    target_ws = req.expected_word_size_bits
    fwd_name = f"{base_name}_evolved_{target_ws}b"
    inv_name = f"{base_name}_evolved_{target_ws}b_inv"

    # Build user prompt
    sections = []

    sections.append(f"""## Original Component
ID: {request.original_component_id}
Role: {req.role}
Current I/O: {sig.input_byte_size or 'variable'}B input, word_size={sig.word_size_bits or 'unknown'}b
Bit operations detected: {', '.join(sig.bit_operations) if sig.bit_operations else 'none'}
Length checks: {sig.length_checks if sig.length_checks else 'none'}

### Source Code:
```python
{request.original_source}
```""")

    # Cryptographic profile section
    profile_lines = []
    if request.preserve_ddt is not None:
        profile_lines.append(f"- DDT max: {request.preserve_ddt} (target: maintain or improve)")
    if request.preserve_lat is not None:
        profile_lines.append(f"- LAT max: {request.preserve_lat} (target: maintain or improve)")
    profile_lines.append(f"- Bijective: {'yes' if request.preserve_bijectivity else 'no'} "
                         f"(must remain: {'yes' if request.preserve_bijectivity else 'no'})")

    if profile_lines:
        sections.append("### Cryptographic Profile (preserve these properties):\n" + "\n".join(profile_lines))

    sections.append(f"""## Target Requirements
- Input: {req.expected_input_bytes} bytes
- Output: {req.expected_output_bytes} bytes
- Word size: {req.expected_word_size_bits} bits
- Architecture: {req.architecture}

## Mismatch Details
Type: {mm.mismatch_type}
Description: {mm.description}
Suggested fix: {mm.suggested_fix}

## Task
Rewrite the component to satisfy the target requirements above.
Provide exactly two functions:
1. `{fwd_name}(data: bytes) -> bytes` — the forward (encryption) function
2. `{inv_name}(data: bytes) -> bytes` — the inverse (decryption) function

Include all necessary constants (lookup tables, permutation arrays, masks) inline.
Do NOT use import statements. Do NOT use markdown fences.""")

    user = "\n\n".join(sections)

    return MUTATION_SYSTEM, user


# ---------------------------------------------------------------------------
# Mutation execution
# ---------------------------------------------------------------------------

def _strip_markdown_fences(text: str) -> str:
    """Remove markdown code fences from LLM output."""
    text = re.sub(r"^```(?:python)?\s*\n?", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n?```\s*$", "", text, flags=re.MULTILINE)
    return text.strip()


def _extract_thinking(text: str) -> Tuple[str, Optional[str]]:
    """Extract DeepSeek-R1 thinking trace from response."""
    reasoning = None
    if "<think>" in text:
        try:
            start = text.index("<think>") + len("<think>")
            end = text.index("</think>") if "</think>" in text else len(text)
            reasoning = text[start:end].strip()
            # Remove thinking block from code
            text = text[:text.index("<think>")] + text[end + len("</think>"):] if "</think>" in text else text[:text.index("<think>")]
        except ValueError:
            pass
    return text.strip(), reasoning


def mutate_component(
    settings: Settings,
    request: MutationRequest,
) -> MutationResult:
    """Call DeepSeek-R1 (or OpenAI fallback) to rewrite a component.

    Args:
        settings: Application settings with API keys.
        request: MutationRequest with all context.

    Returns:
        MutationResult with generated code (not yet compiled/validated).
    """
    system, user = build_mutation_prompt(request)

    provider = OpenAIProvider(
        api_key=settings.openai_api_key,
        openrouter_api_key=settings.openrouter_api_key,
    )

    req = request.target_requirement
    target_ws = req.expected_word_size_bits
    base_name = request.original_component_id.replace(".", "_")
    new_id = f"{request.original_component_id}_evolved_{target_ws}b"
    fwd_name = f"{base_name}_evolved_{target_ws}b"
    inv_name = f"{base_name}_evolved_{target_ws}b_inv"

    reasoning_trace = None

    if provider.openrouter_client and settings.openrouter_api_key:
        model_used = settings.openrouter_model_reasoning
        result = provider.generate_text_chat(
            model=model_used,
            system=system,
            user=user,
            temperature=0.2,
            max_tokens=8192,
            use_openrouter=True,
        )
        code = result.text
        code, reasoning_trace = _extract_thinking(code)
    else:
        model_used = settings.openai_model_code
        result = provider.generate_text(
            model=model_used,
            system=system,
            user=user,
            temperature=0.2,
            max_output_tokens=4096,
        )
        code = result.text

    code = _strip_markdown_fences(code)

    return MutationResult(
        success=bool(code.strip()),
        new_component_id=new_id,
        new_source_code=code,
        new_forward_name=fwd_name,
        new_inverse_name=inv_name,
        reasoning_trace=reasoning_trace,
        model_used=model_used,
    )


def mutate_with_retry(
    settings: Settings,
    request: MutationRequest,
    validation_errors: List[str],
    max_retries: int = 2,
) -> MutationResult:
    """Mutate a component with retry on validation failure.

    On each retry, appends the validation errors to the prompt so the LLM
    can self-correct.

    Args:
        settings: Application settings.
        request: Original mutation request.
        validation_errors: Errors from previous attempt.
        max_retries: Maximum number of retry attempts.

    Returns:
        MutationResult from the last attempt.
    """
    last_result = MutationResult(success=False)

    for attempt in range(max_retries):
        # Build prompt with error feedback
        system, user = build_mutation_prompt(request)

        if validation_errors:
            error_block = "\n".join(f"- {e}" for e in validation_errors)
            user += (
                f"\n\n## Previous Attempt Failed (attempt {attempt + 1}/{max_retries})\n"
                f"The previous code had these validation errors:\n{error_block}\n\n"
                f"Fix these issues in your new version."
            )

        provider = OpenAIProvider(
            api_key=settings.openai_api_key,
            openrouter_api_key=settings.openrouter_api_key,
        )

        req = request.target_requirement
        target_ws = req.expected_word_size_bits
        base_name = request.original_component_id.replace(".", "_")
        new_id = f"{request.original_component_id}_evolved_{target_ws}b"
        fwd_name = f"{base_name}_evolved_{target_ws}b"
        inv_name = f"{base_name}_evolved_{target_ws}b_inv"

        reasoning_trace = None

        if provider.openrouter_client and settings.openrouter_api_key:
            model_used = settings.openrouter_model_reasoning
            result = provider.generate_text_chat(
                model=model_used, system=system, user=user,
                temperature=0.2, max_tokens=8192, use_openrouter=True,
            )
            code = result.text
            code, reasoning_trace = _extract_thinking(code)
        else:
            model_used = settings.openai_model_code
            result = provider.generate_text(
                model=model_used, system=system, user=user,
                temperature=0.2, max_output_tokens=4096,
            )
            code = result.text

        code = _strip_markdown_fences(code)

        last_result = MutationResult(
            success=bool(code.strip()),
            new_component_id=new_id,
            new_source_code=code,
            new_forward_name=fwd_name,
            new_inverse_name=inv_name,
            reasoning_trace=reasoning_trace,
            model_used=model_used,
        )

        if last_result.success:
            break

    return last_result


# ---------------------------------------------------------------------------
# Post-mutation validation
# ---------------------------------------------------------------------------

def validate_mutation(
    result: MutationResult,
    request: MutationRequest,
    compiled_funcs: Dict[str, Any],
    degradation_threshold: float = 0.20,
) -> Tuple[bool, List[str]]:
    """Validate a mutated component's cryptographic properties.

    Args:
        result: MutationResult with generated code.
        request: Original MutationRequest with preservation targets.
        compiled_funcs: Dict of compiled callable functions from dynamic_loader.
        degradation_threshold: Maximum allowed DDT/LAT degradation fraction.

    Returns:
        Tuple of (passed: bool, errors: List[str]).
    """
    errors: List[str] = []
    req = request.target_requirement

    fwd = compiled_funcs.get(result.new_forward_name)
    inv = compiled_funcs.get(result.new_inverse_name)

    if fwd is None:
        errors.append(f"Forward function '{result.new_forward_name}' not found in compiled output")
        return False, errors

    if inv is None and request.preserve_bijectivity:
        errors.append(f"Inverse function '{result.new_inverse_name}' not found but bijectivity required")
        return False, errors

    # Test 1: I/O size check
    import random
    rng = random.Random(42)
    test_input = bytes(rng.randrange(0, 256) for _ in range(req.expected_input_bytes))

    try:
        output = fwd(test_input)
        if len(output) != req.expected_output_bytes:
            errors.append(
                f"Forward output size {len(output)} != expected {req.expected_output_bytes}"
            )
    except Exception as e:
        errors.append(f"Forward function crashed on {req.expected_input_bytes}-byte input: {e}")
        return False, errors

    # Test 2: Bijectivity (forward then inverse = identity)
    if request.preserve_bijectivity and inv is not None:
        for _ in range(50):
            test_data = bytes(rng.randrange(0, 256) for _ in range(req.expected_input_bytes))
            try:
                encrypted = fwd(test_data)
                decrypted = inv(encrypted)
                if decrypted != test_data:
                    errors.append(
                        f"Bijectivity failed: inv(fwd(x)) != x for input {test_data.hex()[:16]}..."
                    )
                    break
            except Exception as e:
                errors.append(f"Bijectivity test crashed: {e}")
                break

    # Test 3: Cryptographic profile check (for S-boxes only)
    if request.preserve_ddt is not None and req.role in ("sbox", "f_sbox"):
        try:
            from cipherlab.cipher.cryptanalysis import sbox_ddt_max, sbox_lat_max_abs

            # Determine S-box size
            is_4bit = any(tag in request.original_component_id for tag in ("present", "gift"))
            sbox_size = 16 if is_4bit else 256

            # Extract lookup table
            table = []
            for val in range(sbox_size):
                inp = bytes([val])
                out = fwd(inp)
                table.append(out[0] & 0x0F if is_4bit else out[0])

            new_ddt = sbox_ddt_max(table)
            if request.preserve_ddt is not None:
                max_allowed = int(request.preserve_ddt * (1 + degradation_threshold))
                if new_ddt > max_allowed:
                    errors.append(
                        f"DDT degradation: new DDT_max={new_ddt} exceeds "
                        f"threshold {max_allowed} (original={request.preserve_ddt})"
                    )

            new_lat = sbox_lat_max_abs(table)
            if request.preserve_lat is not None:
                max_allowed_lat = int(request.preserve_lat * (1 + degradation_threshold))
                if new_lat > max_allowed_lat:
                    errors.append(
                        f"LAT degradation: new LAT_max={new_lat} exceeds "
                        f"threshold {max_allowed_lat} (original={request.preserve_lat})"
                    )
        except Exception as e:
            errors.append(f"Cryptographic profile check failed: {e}")

    return len(errors) == 0, errors
