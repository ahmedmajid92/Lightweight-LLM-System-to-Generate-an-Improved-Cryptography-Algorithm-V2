from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from ..config import Settings
from ..rag.retriever import RAGRetriever
from ..cipher.spec import CipherSpec, ImprovementPatch
from .openai_provider import OpenAIProvider


SYSTEM_IMPROVER = """You are a cryptography research assistant focused on LIGHTWEIGHT BLOCK CIPHERS for IoT and resource-constrained environments.
You help improve *research prototype* ciphers constructed from components (SPN, Feistel, or ARX).

The 12 baseline LWC algorithms are: AES, DES, SIMON, SPECK, PRESENT, TEA, XTEA, RC5, Blowfish, HIGHT, LEA, GIFT.

Rules:
- Do NOT claim security. You can recommend design changes, but must phrase as hypotheses.
- Prefer changes that improve diffusion/confusion while keeping implementation lightweight and efficient.
- Consider hardware gate count, power consumption, and memory footprint as design constraints.
- Respect the cipher architecture constraints.
- Output must match the provided schema exactly (Structured Outputs).
"""


def suggest_improvements(
    *,
    settings: Settings,
    spec: CipherSpec,
    metrics: Dict[str, object],
    issues: List[str],
    rag_context: str,
    model: str,
    openrouter_model: Optional[str] = None,
) -> tuple[ImprovementPatch, Any, str]:
    provider = OpenAIProvider(
        api_key=settings.openai_api_key,
        openrouter_api_key=settings.openrouter_api_key,
        local_reasoning_base_url=settings.local_reasoning_base_url if settings.local_llm_enabled else None,
        local_api_key=settings.local_llm_api_key,
        local_timeout_seconds=settings.local_llm_timeout_seconds,
    )

    user = (
        "Current cipher spec (JSON):\n"
        + spec.model_dump_json(indent=2)
        + "\n\nLocal metrics (JSON):\n"
        + json.dumps(metrics, indent=2)
        + "\n\nDetected issues:\n- "
        + "\n- ".join(issues or ["(none)"])
        + "\n\nRelevant KB context:\n"
        + rag_context
        + "\n\nTask: propose a small ImprovementPatch that is likely to improve avalanche/diffusion. "
        + "You may suggest: increase rounds, replace sbox/perm/linear, or add notes. "
        + "Do not suggest components that are not available in the registry IDs shown below.\n\n"
        + "Available component IDs:\n"
        + "- key_schedule: ks.sha256_kdf\n"
        + "- sbox: sbox.aes, sbox.identity\n"
        + "- perm: perm.aes_shiftrows, perm.identity\n"
        + "- linear: linear.aes_mixcolumns, linear.identity\n"
        + "- f_sbox: sbox.aes, sbox.identity\n"
        + "- f_perm: perm.identity\n"
    )

    patch, raw, model_used = provider.generate_structured_with_fallback(
        openrouter_model=openrouter_model,
        fallback_model=model,
        local_model=settings.local_reasoning_model if settings.local_llm_enabled else None,
        local_role="reasoning",
        system=SYSTEM_IMPROVER,
        user=user,
        schema=ImprovementPatch,
        primary_temperature=0.2,
        primary_max_tokens=4096,
        fallback_temperature=0.2,
        fallback_max_output_tokens=800,
    )
    return patch, raw, model_used
