from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Optional, Type, TypeVar

from pydantic import BaseModel

try:
    from openai import OpenAI
except Exception as e:  # pragma: no cover
    OpenAI = None  # type: ignore


T = TypeVar("T", bound=BaseModel)


@dataclass
class LLMUsage:
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None


@dataclass
class LLMResult:
    model: str
    text: str
    raw: Any
    usage: LLMUsage


class OpenAIProvider:
    """Wrapper around the OpenAI Python SDK supporting dual clients.

    * Standard OpenAI client — uses the Responses API for chat and embeddings.
    * OpenRouter client — uses the Chat Completions API for DeepSeek
      code-generation / reasoning models.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        openrouter_api_key: Optional[str] = None,
    ):
        if OpenAI is None:
            raise RuntimeError("openai package not installed. pip install -r requirements.txt")
        # Standard OpenAI client (for chat, embeddings)
        self.client = OpenAI(api_key=api_key)
        # OpenRouter client (for code-generation / reasoning via DeepSeek)
        self.openrouter_client: Optional[OpenAI] = None
        if openrouter_api_key:
            self.openrouter_client = OpenAI(
                api_key=openrouter_api_key,
                base_url="https://openrouter.ai/api/v1",
            )

    # ------------------------------------------------------------------
    # OpenAI Responses API methods (unchanged)
    # ------------------------------------------------------------------

    def generate_text(
        self,
        *,
        model: str,
        system: str,
        user: str,
        temperature: float = 0.2,
        max_output_tokens: int = 1200,
    ) -> LLMResult:
        resp = self.client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )
        usage = LLMUsage(
            input_tokens=getattr(resp.usage, "input_tokens", None) if getattr(resp, "usage", None) else None,
            output_tokens=getattr(resp.usage, "output_tokens", None) if getattr(resp, "usage", None) else None,
        )
        text = getattr(resp, "output_text", "") or ""
        return LLMResult(model=model, text=text, raw=resp, usage=usage)

    def generate_structured(
        self,
        *,
        model: str,
        system: str,
        user: str,
        schema: Type[T],
        temperature: float = 0.0,
        max_output_tokens: int = 1200,
    ) -> tuple[T, Any]:
        """Return a schema-validated object using Structured Outputs.

        Requires a recent OpenAI SDK version that supports `responses.parse`.
        """
        resp = self.client.responses.parse(
            model=model,
            input=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            text_format=schema,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )
        parsed = resp.output_parsed
        return parsed, resp

    def generate_json_fallback(
        self,
        *,
        model: str,
        system: str,
        user: str,
        temperature: float = 0.0,
        max_output_tokens: int = 1200,
        retries: int = 2,
    ) -> tuple[dict, Any]:
        """Fallback if you don't want Structured Outputs.

        Uses plain text generation and parses JSON with basic retries.
        """
        prompt = (
            user
            + "\n\nReturn ONLY valid JSON. Do not wrap in markdown. Do not include commentary."
        )
        last_err: Optional[Exception] = None
        last_resp: Any = None
        for _ in range(retries + 1):
            r = self.generate_text(
                model=model, system=system, user=prompt, temperature=temperature, max_output_tokens=max_output_tokens
            )
            last_resp = r.raw
            try:
                return json.loads(r.text), last_resp
            except Exception as e:
                last_err = e
        raise ValueError(f"Failed to parse JSON from model output: {last_err}")

    # ------------------------------------------------------------------
    # Chat Completions API methods (OpenRouter-compatible)
    # ------------------------------------------------------------------

    def generate_text_chat(
        self,
        *,
        model: str,
        system: str,
        user: str,
        temperature: float = 0.2,
        max_tokens: int = 4096,
        use_openrouter: bool = False,
    ) -> LLMResult:
        """Generate text via the Chat Completions API (OpenRouter-compatible)."""
        client = self.openrouter_client if use_openrouter and self.openrouter_client else self.client
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        choice = resp.choices[0]
        usage = LLMUsage(
            input_tokens=getattr(resp.usage, "prompt_tokens", None) if resp.usage else None,
            output_tokens=getattr(resp.usage, "completion_tokens", None) if resp.usage else None,
        )
        return LLMResult(model=model, text=choice.message.content or "", raw=resp, usage=usage)

    def generate_json_chat(
        self,
        *,
        model: str,
        system: str,
        user: str,
        schema: Type[T],
        temperature: float = 0.0,
        max_tokens: int = 4096,
        retries: int = 2,
        use_openrouter: bool = False,
    ) -> tuple[T, Any]:
        """Generate structured output via Chat Completions + JSON parsing.

        OpenRouter-compatible: appends the JSON schema to the system prompt
        and validates the response with Pydantic.
        """
        client = self.openrouter_client if use_openrouter and self.openrouter_client else self.client
        schema_json = json.dumps(schema.model_json_schema(), indent=2)
        augmented_system = (
            system
            + "\n\nYou MUST respond with valid JSON matching this exact schema:\n"
            + schema_json
        )
        last_err: Optional[Exception] = None
        for _ in range(retries + 1):
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": augmented_system},
                    {"role": "user", "content": user + "\n\nRespond ONLY with valid JSON. No markdown fences."},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            text = resp.choices[0].message.content or ""
            # Strip markdown code fences if the model wraps its output
            text = re.sub(r"^```(?:json)?\s*", "", text.strip())
            text = re.sub(r"\s*```$", "", text.strip())
            try:
                parsed = schema.model_validate_json(text)
                return parsed, resp
            except Exception as e:
                last_err = e
        raise ValueError(f"Failed to parse structured output after {retries + 1} tries: {last_err}")
