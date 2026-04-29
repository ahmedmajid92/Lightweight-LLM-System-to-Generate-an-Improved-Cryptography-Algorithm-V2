from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Optional, Type, TypeVar

from pydantic import BaseModel

try:
    from openai import OpenAI
except Exception as e:  # pragma: no cover
    OpenAI = None  # type: ignore


T = TypeVar("T", bound=BaseModel)
logger = logging.getLogger("cipherlab.llm")


_CONNECTION_ERROR_NAMES = {
    "APIConnectionError",
    "APITimeoutError",
    "ConnectError",
    "ConnectTimeout",
    "ConnectionError",
    "MaxRetryError",
    "NewConnectionError",
    "ReadError",
    "ReadTimeout",
    "Timeout",
    "TimeoutError",
    "TimeoutException",
}


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
        local_reasoning_base_url: Optional[str] = None,
        local_code_base_url: Optional[str] = None,
        local_api_key: str = "sk-no-key-required",
        local_timeout_seconds: float = 60.0,
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
        self.local_reasoning_client: Optional[OpenAI] = None
        self.local_code_client: Optional[OpenAI] = None
        if local_reasoning_base_url:
            self.local_reasoning_client = OpenAI(
                api_key=local_api_key,
                base_url=local_reasoning_base_url,
                timeout=local_timeout_seconds,
            )
        if local_code_base_url:
            self.local_code_client = OpenAI(
                api_key=local_api_key,
                base_url=local_code_base_url,
                timeout=local_timeout_seconds,
            )

    # ------------------------------------------------------------------
    # OpenAI Responses API methods (unchanged)
    # ------------------------------------------------------------------

    @staticmethod
    def _is_unsupported_temperature_error(exc: Exception) -> bool:
        text = str(exc)
        return "Unsupported parameter" in text and "'temperature'" in text

    @staticmethod
    def _is_connection_failure(exc: Exception) -> bool:
        current: Optional[BaseException] = exc
        while current is not None:
            if current.__class__.__name__ in _CONNECTION_ERROR_NAMES:
                return True
            if isinstance(current, (ConnectionError, TimeoutError)):
                return True
            current = current.__cause__ or current.__context__
        text = str(exc).lower()
        return any(
            marker in text
            for marker in (
                "connection error",
                "connection refused",
                "connection reset",
                "connection aborted",
                "could not resolve",
                "dns",
                "failed to establish",
                "network is unreachable",
                "timed out",
                "timeout",
            )
        )

    @staticmethod
    def _schema_augmented_system(system: str, schema: Type[T]) -> str:
        schema_json = json.dumps(schema.model_json_schema(), indent=2)
        required_fields = schema.model_json_schema().get("required", [])
        field_hint = ""
        if required_fields:
            field_hint = (
                "\n\nREQUIRED top-level JSON keys (use these EXACT names): "
                + ", ".join(f'"{f}"' for f in required_fields)
            )
        return (
            system
            + "\n\nYou MUST respond with valid JSON matching this exact schema:\n"
            + schema_json
            + field_hint
        )

    @staticmethod
    def _clean_json_text(text: str) -> str:
        text = text.strip()
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        return text.strip()

    @staticmethod
    def _local_client_name(role: str) -> str:
        return "local_reasoning_client" if role == "reasoning" else "local_code_client"

    def _get_local_client(self, role: str) -> Optional[Any]:
        return getattr(self, self._local_client_name(role), None)

    def _responses_create_with_temperature_retry(self, **kwargs: Any) -> Any:
        try:
            return self.client.responses.create(**kwargs)
        except Exception as exc:
            if "temperature" in kwargs and self._is_unsupported_temperature_error(exc):
                retry_kwargs = dict(kwargs)
                retry_kwargs.pop("temperature", None)
                return self.client.responses.create(**retry_kwargs)
            raise

    def _responses_parse_with_temperature_retry(self, **kwargs: Any) -> Any:
        try:
            return self.client.responses.parse(**kwargs)
        except Exception as exc:
            if "temperature" in kwargs and self._is_unsupported_temperature_error(exc):
                retry_kwargs = dict(kwargs)
                retry_kwargs.pop("temperature", None)
                return self.client.responses.parse(**retry_kwargs)
            raise

    def generate_text(
        self,
        *,
        model: str,
        system: str,
        user: str,
        temperature: float = 0.2,
        max_output_tokens: int = 1200,
    ) -> LLMResult:
        resp = self._responses_create_with_temperature_retry(
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
        resp = self._responses_parse_with_temperature_retry(
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
        augmented_system = self._schema_augmented_system(system, schema)
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
            text = self._clean_json_text(resp.choices[0].message.content or "")
            try:
                parsed = schema.model_validate_json(text)
                return parsed, resp
            except Exception as e:
                last_err = e
        raise ValueError(f"Failed to parse structured output after {retries + 1} tries: {last_err}")

    def generate_text_local(
        self,
        *,
        role: str,
        model: str,
        system: str,
        user: str,
        temperature: float = 0.2,
        max_tokens: int = 4096,
    ) -> LLMResult:
        """Generate text from a local llama.cpp OpenAI-compatible server."""
        client = self._get_local_client(role)
        if client is None:
            raise RuntimeError(f"Local {role} client is not configured")
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
            input_tokens=getattr(resp.usage, "prompt_tokens", None) if getattr(resp, "usage", None) else None,
            output_tokens=getattr(resp.usage, "completion_tokens", None) if getattr(resp, "usage", None) else None,
        )
        return LLMResult(model=model, text=choice.message.content or "", raw=resp, usage=usage)

    def generate_structured_local(
        self,
        *,
        role: str,
        model: str,
        system: str,
        user: str,
        schema: Type[T],
        temperature: float = 0.0,
        max_tokens: int = 4096,
        retries: int = 2,
    ) -> tuple[T, Any]:
        """Generate schema-validated JSON from a local llama.cpp server."""
        client = self._get_local_client(role)
        if client is None:
            raise RuntimeError(f"Local {role} client is not configured")
        augmented_system = self._schema_augmented_system(system, schema)
        last_err: Optional[Exception] = None
        last_resp: Any = None
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
            last_resp = resp
            text = self._clean_json_text(resp.choices[0].message.content or "")
            try:
                return schema.model_validate_json(text), resp
            except Exception as exc:
                last_err = exc
        raise ValueError(f"Failed to parse local structured output after {retries + 1} tries: {last_err}") from last_err

    def generate_text_with_fallback(
        self,
        *,
        openrouter_model: Optional[str],
        fallback_model: str,
        system: str,
        user: str,
        local_model: Optional[str] = None,
        local_role: str = "reasoning",
        primary_temperature: float = 0.2,
        primary_max_tokens: int = 4096,
        fallback_temperature: float = 0.2,
        fallback_max_output_tokens: int = 1200,
        local_temperature: Optional[float] = None,
        local_max_tokens: Optional[int] = None,
    ) -> tuple[LLMResult, str]:
        """Use OpenRouter first, then fall back to the standard OpenAI client on failure."""
        primary_err: Optional[Exception] = None
        if self.openrouter_client and openrouter_model:
            try:
                result = self.generate_text_chat(
                    model=openrouter_model,
                    system=system,
                    user=user,
                    temperature=primary_temperature,
                    max_tokens=primary_max_tokens,
                    use_openrouter=True,
                )
                return result, openrouter_model
            except Exception as exc:
                primary_err = exc

        try:
            result = self.generate_text(
                model=fallback_model,
                system=system,
                user=user,
                temperature=fallback_temperature,
                max_output_tokens=fallback_max_output_tokens,
            )
            return result, fallback_model
        except Exception as fallback_err:
            if local_model and self._is_connection_failure(fallback_err):
                logger.warning(
                    "[llm-fallback] Remote %s call unavailable: %s. Using local %s.",
                    local_role,
                    fallback_err,
                    local_model,
                )
                result = self.generate_text_local(
                    role=local_role,
                    model=local_model,
                    system=system,
                    user=user,
                    temperature=fallback_temperature if local_temperature is None else local_temperature,
                    max_tokens=fallback_max_output_tokens if local_max_tokens is None else local_max_tokens,
                )
                return result, local_model
            if primary_err is not None:
                raise RuntimeError(
                    f"OpenRouter model '{openrouter_model}' failed and fallback model '{fallback_model}' also failed. "
                    f"Primary error: {primary_err}; fallback error: {fallback_err}"
                ) from fallback_err
            raise

    def generate_structured_with_fallback(
        self,
        *,
        openrouter_model: Optional[str],
        fallback_model: str,
        system: str,
        user: str,
        schema: Type[T],
        local_model: Optional[str] = None,
        local_role: str = "reasoning",
        primary_temperature: float = 0.0,
        primary_max_tokens: int = 4096,
        fallback_temperature: float = 0.0,
        fallback_max_output_tokens: int = 1200,
        local_temperature: Optional[float] = None,
        local_max_tokens: Optional[int] = None,
    ) -> tuple[T, Any, str]:
        """Use OpenRouter JSON generation first, then fall back to OpenAI Structured Outputs."""
        primary_err: Optional[Exception] = None
        if self.openrouter_client and openrouter_model:
            try:
                parsed, raw = self.generate_json_chat(
                    model=openrouter_model,
                    system=system,
                    user=user,
                    schema=schema,
                    temperature=primary_temperature,
                    max_tokens=primary_max_tokens,
                    use_openrouter=True,
                )
                return parsed, raw, openrouter_model
            except Exception as exc:
                primary_err = exc

        try:
            parsed, raw = self.generate_structured(
                model=fallback_model,
                system=system,
                user=user,
                schema=schema,
                temperature=fallback_temperature,
                max_output_tokens=fallback_max_output_tokens,
            )
            return parsed, raw, fallback_model
        except Exception as fallback_err:
            if local_model and self._is_connection_failure(fallback_err):
                logger.warning(
                    "[llm-fallback] Remote %s call unavailable: %s. Using local %s.",
                    local_role,
                    fallback_err,
                    local_model,
                )
                parsed, raw = self.generate_structured_local(
                    role=local_role,
                    model=local_model,
                    system=system,
                    user=user,
                    schema=schema,
                    temperature=fallback_temperature if local_temperature is None else local_temperature,
                    max_tokens=fallback_max_output_tokens if local_max_tokens is None else local_max_tokens,
                )
                return parsed, raw, local_model
            if primary_err is not None:
                raise RuntimeError(
                    f"OpenRouter model '{openrouter_model}' failed and fallback model '{fallback_model}' also failed. "
                    f"Primary error: {primary_err}; fallback error: {fallback_err}"
                ) from fallback_err
            raise
