from __future__ import annotations

import json
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
    """Minimal wrapper around the OpenAI Python SDK (Responses API).

    Uses `client.responses.create` for plain text, and `client.responses.parse`
    for Structured Outputs when you pass a Pydantic model.
    """

    def __init__(self, api_key: Optional[str] = None):
        if OpenAI is None:
            raise RuntimeError("openai package not installed. pip install -r requirements.txt")
        self.client = OpenAI(api_key=api_key)

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
