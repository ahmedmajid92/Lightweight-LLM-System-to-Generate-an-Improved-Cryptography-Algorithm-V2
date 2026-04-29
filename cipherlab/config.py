from __future__ import annotations

import os
from functools import lru_cache
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field


class Settings(BaseModel):
    # OpenAI
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key")
    openai_model_fast: str = Field(default="gpt-5-mini")
    openai_model_quality: str = Field(default="gpt-5.2")
    openai_model_code: str = Field(default="gpt-5.2-codex")
    openai_embedding_model: str = Field(default="text-embedding-3-small")

    # OpenRouter (DeepSeek)
    openrouter_api_key: Optional[str] = Field(default=None, description="OpenRouter API key")
    openrouter_model_fast: str = Field(default="deepseek/deepseek-chat-v3-0324")
    openrouter_model_reasoning: str = Field(default="deepseek/deepseek-r1")

    # Local llama.cpp fallbacks
    local_llm_enabled: bool = Field(default=False)
    local_llm_api_key: str = Field(default="sk-no-key-required")
    local_llm_timeout_seconds: float = Field(default=60.0, ge=1.0)
    local_reasoning_base_url: str = Field(default="http://127.0.0.1:8081/v1")
    local_reasoning_model: str = Field(default="DeepSeek-R1-Distill-Qwen-7B")
    local_code_base_url: str = Field(default="http://127.0.0.1:8082/v1")
    local_code_model: str = Field(default="Qwen2.5-Coder-7B-Instruct")

    # RAG
    rag_use_embeddings: bool = Field(default=True)
    rag_embedding_provider: str = Field(default="auto")
    rag_hybrid_alpha: float = Field(default=0.60, ge=0.0, le=1.0)
    rag_bm25_weight: float = Field(default=0.40, ge=0.0, le=1.0)
    rag_top_k: int = Field(default=6, ge=1, le=20)
    rag_local_embeddings_enabled: bool = Field(default=True)
    rag_local_embedding_base_url: str = Field(default="http://127.0.0.1:8083/v1")
    rag_local_embedding_model: str = Field(default="nomic-embed-text-v1.5")
    rag_local_embedding_dim: int = Field(default=512, ge=64, le=768)
    rag_local_embedding_timeout_seconds: float = Field(default=60.0, ge=1.0)
    rag_rerank_enabled: bool = Field(default=False)
    rag_rerank_base_url: str = Field(default="http://127.0.0.1:8084/v1")
    rag_rerank_model: str = Field(default="Qwen3-Reranker-0.6B")
    rag_rerank_top_n: int = Field(default=30, ge=1, le=100)
    rag_final_top_k: int = Field(default=6, ge=1, le=20)

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

    rag_dense_weight = float(os.getenv("RAG_DENSE_WEIGHT", os.getenv("RAG_HYBRID_ALPHA", "0.60")))
    rag_final_top_k = int(os.getenv("RAG_FINAL_TOP_K", os.getenv("RAG_TOP_K", "6")))

    return Settings(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_model_fast=os.getenv("OPENAI_MODEL_FAST", "gpt-5-mini"),
        openai_model_quality=os.getenv("OPENAI_MODEL_QUALITY", "gpt-5.2"),
        openai_model_code=os.getenv("OPENAI_MODEL_CODE", "gpt-5.2-codex"),
        openai_embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        openrouter_api_key=os.getenv("OPENROUTER_API_KEY"),
        openrouter_model_fast=os.getenv("OPENROUTER_MODEL_FAST", "deepseek/deepseek-chat-v3-0324"),
        openrouter_model_reasoning=os.getenv("OPENROUTER_MODEL_REASONING", "deepseek/deepseek-r1"),
        local_llm_enabled=_bool("LOCAL_LLM_ENABLED", False),
        local_llm_api_key=os.getenv("LOCAL_LLM_API_KEY", "sk-no-key-required"),
        local_llm_timeout_seconds=float(os.getenv("LOCAL_LLM_TIMEOUT_SECONDS", "60")),
        local_reasoning_base_url=os.getenv("LOCAL_REASONING_BASE_URL", "http://127.0.0.1:8081/v1"),
        local_reasoning_model=os.getenv("LOCAL_REASONING_MODEL", "DeepSeek-R1-Distill-Qwen-7B"),
        local_code_base_url=os.getenv("LOCAL_CODE_BASE_URL", "http://127.0.0.1:8082/v1"),
        local_code_model=os.getenv("LOCAL_CODE_MODEL", "Qwen2.5-Coder-7B-Instruct"),
        rag_use_embeddings=_bool("RAG_USE_EMBEDDINGS", True),
        rag_embedding_provider=os.getenv("RAG_EMBEDDING_PROVIDER", "auto").strip().lower(),
        rag_hybrid_alpha=rag_dense_weight,
        rag_bm25_weight=float(os.getenv("RAG_BM25_WEIGHT", str(max(0.0, 1.0 - rag_dense_weight)))),
        rag_top_k=rag_final_top_k,
        rag_local_embeddings_enabled=_bool("RAG_LOCAL_EMBEDDINGS_ENABLED", True),
        rag_local_embedding_base_url=os.getenv("RAG_LOCAL_EMBEDDING_BASE_URL", "http://127.0.0.1:8083/v1"),
        rag_local_embedding_model=os.getenv("RAG_LOCAL_EMBEDDING_MODEL", "nomic-embed-text-v1.5"),
        rag_local_embedding_dim=int(os.getenv("RAG_LOCAL_EMBED_DIM", "512")),
        rag_local_embedding_timeout_seconds=float(os.getenv("RAG_LOCAL_EMBEDDING_TIMEOUT_SECONDS", "60")),
        rag_rerank_enabled=_bool("RAG_RERANK_ENABLED", False),
        rag_rerank_base_url=os.getenv("RAG_RERANK_BASE_URL", "http://127.0.0.1:8084/v1"),
        rag_rerank_model=os.getenv("RAG_RERANK_MODEL", "Qwen3-Reranker-0.6B"),
        rag_rerank_top_n=int(os.getenv("RAG_RERANK_TOP_N", "30")),
        rag_final_top_k=rag_final_top_k,
        global_seed=int(os.getenv("GLOBAL_SEED", "1337")),
    )
