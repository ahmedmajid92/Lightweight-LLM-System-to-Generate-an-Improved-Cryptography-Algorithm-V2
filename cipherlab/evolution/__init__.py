"""Adaptive Structural Evolution for cipher components.

Provides AST-based dependency mapping, autonomous component mutation
via DeepSeek-R1, and safe dynamic re-importation into the live registry.

Research / education only. Do NOT use in production.
"""

from .ast_analyzer import (
    DataFlowSignature,
    ComponentRequirement,
    MismatchReport,
    analyze_component_code,
    derive_requirements,
    detect_mismatches,
    analyze_llm_cipher_code,
)
from .component_mutator import (
    MutationRequest,
    MutationResult,
    build_mutation_prompt,
    mutate_component,
    validate_mutation,
)
from .dynamic_loader import (
    LoadedComponent,
    EvolutionReport,
    compile_component_code,
    inject_component,
    evolve_component,
    evolve_all_mismatches,
)

__all__ = [
    "DataFlowSignature",
    "ComponentRequirement",
    "MismatchReport",
    "analyze_component_code",
    "derive_requirements",
    "detect_mismatches",
    "analyze_llm_cipher_code",
    "MutationRequest",
    "MutationResult",
    "build_mutation_prompt",
    "mutate_component",
    "validate_mutation",
    "LoadedComponent",
    "EvolutionReport",
    "compile_component_code",
    "inject_component",
    "evolve_component",
    "evolve_all_mismatches",
]
