from __future__ import annotations

import os
from functools import lru_cache
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field


class Settings(BaseModel):
    # OpenAI
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key")
    openai_model_fast: str = Field(default="gpt-4.1-mini")
    openai_model_quality: str = Field(default="gpt-5.2")
    openai_model_code: str = Field(default="gpt-5.1-codex")
    openai_embedding_model: str = Field(default="text-embedding-3-small")

    # OpenRouter (DeepSeek)
    openrouter_api_key: Optional[str] = Field(default=None, description="OpenRouter API key")
    openrouter_model_fast: str = Field(default="deepseek/deepseek-chat-v3-0324")
    openrouter_model_reasoning: str = Field(default="deepseek/deepseek-r1")

    # RAG
    rag_use_embeddings: bool = Field(default=True)
    rag_hybrid_alpha: float = Field(default=0.55, ge=0.0, le=1.0)
    rag_top_k: int = Field(default=6, ge=1, le=20)

    # Reproducibility
    global_seed: int = Field(default=1337)

    # Paths
    project_root: str = Field(default=os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
    kb_dir: str = Field(default="kb")
    kb_external_dir: str = Field(default="kb_external")
    kb_index_dir: str = Field(default="kb_index")
    runs_dir: str = Field(default="runs")


@lru_cache(maxsize=1)
def load_settings() -> Settings:
    # Load .env if present
    load_dotenv()

    def _bool(name: str, default: bool) -> bool:
        v = os.getenv(name)
        if v is None:
            return default
        return v.strip().lower() in {"1", "true", "yes", "y", "on"}

    return Settings(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_model_fast=os.getenv("OPENAI_MODEL_FAST", "gpt-5-mini"),
        openai_model_quality=os.getenv("OPENAI_MODEL_QUALITY", "gpt-5.2"),
        openai_model_code=os.getenv("OPENAI_MODEL_CODE", "gpt-5.1-codex"),
        openai_embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        openrouter_api_key=os.getenv("OPENROUTER_API_KEY"),
        openrouter_model_fast=os.getenv("OPENROUTER_MODEL_FAST", "deepseek/deepseek-chat-v3-0324"),
        openrouter_model_reasoning=os.getenv("OPENROUTER_MODEL_REASONING", "deepseek/deepseek-r1"),
        rag_use_embeddings=_bool("RAG_USE_EMBEDDINGS", True),
        rag_hybrid_alpha=float(os.getenv("RAG_HYBRID_ALPHA", "0.55")),
        rag_top_k=int(os.getenv("RAG_TOP_K", "6")),
        global_seed=int(os.getenv("GLOBAL_SEED", "1337")),
    )
