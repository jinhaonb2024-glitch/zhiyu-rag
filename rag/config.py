"""Explicit settings for independent OpenAI-compatible chat and embedding providers."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Settings:
    mode: str = "demo"
    llm_api_key: str = field(default="", repr=False)
    llm_base_url: str = "https://api.openai.com/v1"
    llm_model: str = "gpt-4o-mini"
    embedding_api_key: str = field(default="", repr=False)
    embedding_base_url: str = "https://api.openai.com/v1"
    embedding_model: str = "text-embedding-3-small"
    embedding_batch_size: int = 16
    api_timeout: float = 60
    chunk_size: int = 600
    chunk_overlap: int = 100
    top_k: int = 4
    min_similarity: float = 0.12
    max_context_chars: int = 10000

    @classmethod
    def from_env(cls) -> Settings:
        load_dotenv(ROOT / ".env", override=False)
        common_key = os.getenv("OPENAI_API_KEY", "")
        common_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        return cls(
            mode=os.getenv("RAG_MODE", "demo").strip().lower(),
            llm_api_key=os.getenv("LLM_API_KEY", "").strip() or common_key,
            llm_base_url=os.getenv("LLM_BASE_URL", "").strip() or common_url,
            llm_model=os.getenv("LLM_MODEL", "gpt-4o-mini").strip(),
            embedding_api_key=os.getenv("EMBEDDING_API_KEY", "").strip() or common_key,
            embedding_base_url=os.getenv("EMBEDDING_BASE_URL", "").strip() or common_url,
            embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small").strip(),
            embedding_batch_size=int(os.getenv("EMBEDDING_BATCH_SIZE", "16")),
            api_timeout=float(os.getenv("API_TIMEOUT", "60")),
            chunk_size=int(os.getenv("CHUNK_SIZE", "600")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "100")),
            top_k=int(os.getenv("TOP_K", "4")),
            min_similarity=float(os.getenv("MIN_SIMILARITY", "0.12")),
            max_context_chars=int(os.getenv("MAX_CONTEXT_CHARS", "10000")),
        )

    def validate(self, *, embeddings: bool = False, chat: bool = False) -> None:
        if self.mode not in {"demo", "api"}:
            raise ValueError("RAG_MODE 只能是 demo 或 api。")
        if not 100 <= self.chunk_size <= 4000:
            raise ValueError("分段长度须在 100～4000 字符之间。")
        if not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError("重叠长度须大于等于 0 且小于分段长度。")
        if not 1 <= self.top_k <= 20:
            raise ValueError("检索片段数量须在 1～20 之间。")
        if not -1 <= self.min_similarity <= 1:
            raise ValueError("相似度阈值须在 -1～1 之间。")
        if self.max_context_chars < 100 or self.embedding_batch_size < 1 or self.api_timeout <= 0:
            raise ValueError("上下文长度、Embedding 批量大小或请求超时配置不正确。")
        if self.mode == "api":
            for required, label, key, url, model in (
                (
                    embeddings,
                    "Embedding",
                    self.embedding_api_key,
                    self.embedding_base_url,
                    self.embedding_model,
                ),
                (chat, "问答模型", self.llm_api_key, self.llm_base_url, self.llm_model),
            ):
                if required and (not key or not model):
                    raise ValueError(f"请先配置{label}的 API Key 和模型名。")
                parsed = urlparse(url)
                if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                    raise ValueError(f"{label} Base URL 须是完整的 http(s) 地址。")
                if parsed.hostname == "cloud.siliconflow.cn":
                    raise ValueError(
                        f"{label} Base URL 填成了硅基流动控制台网页。"
                        "请改为 https://api.siliconflow.cn/v1。"
                    )
                if parsed.hostname in {"platform.deepseek.com", "platform.openai.com"}:
                    raise ValueError(f"{label} Base URL 应填写 API 地址，不能填写平台控制台网页。")
                if parsed.query or parsed.fragment or parsed.username or parsed.password:
                    raise ValueError(f"{label} Base URL 不能包含查询参数、锚点或登录凭据。")
                path = parsed.path.rstrip("/")
                if path.endswith(("/embeddings", "/chat/completions", "/responses", "/models")):
                    raise ValueError(
                        f"{label} Base URL 应填写接口前缀，"
                        "不要包含 /embeddings、/chat/completions、/responses 或 /models。"
                    )

    @property
    def index_signature(self) -> dict:
        """Credentials and chat settings never become part of a persisted index."""
        embedding = (
            {"kind": "demo-char-ngram-v1", "dimensions": 2048}
            if self.mode == "demo"
            else {
                "kind": "openai-compatible-v1",
                "model": self.embedding_model,
                "base_url": self.embedding_base_url.rstrip("/"),
            }
        )
        return {
            "embedding": embedding,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "preprocessing": 1,
        }

    @property
    def index_path(self) -> Path:
        return ROOT / "data" / "indices" / self.mode / "knowledge.npz"
