"""LangChain providers, with an honest deterministic offline demonstration mode."""

from __future__ import annotations

import hashlib
import re
from json import JSONDecodeError

import numpy as np
from langchain_core.embeddings import Embeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from openai import APIResponseValidationError

from rag.config import Settings


class EmbeddingResponseError(ValueError):
    """The configured endpoint returned a webpage or a malformed embedding response."""


INVALID_EMBEDDING_RESPONSE = (
    "Embedding 接口未返回有效的向量数据，可能返回了网页或不兼容的响应。"
    "请检查 Base URL 是否为 API 接口前缀；"
    "硅基流动应填写 https://api.siliconflow.cn/v1。"
)


class DemoEmbeddings(Embeddings):
    """Hashed Chinese character n-grams and English words; NOT semantic embeddings."""

    dimensions = 2048

    def embed_query(self, text: str) -> list[float]:
        vector = np.zeros(self.dimensions, dtype=np.float32)
        words = re.findall(r"[a-z0-9_]+", text.lower())
        for run in re.findall(r"[\u3400-\u9fff]+", text):
            for size in (2, 3):
                words.extend(run[i : i + size] for i in range(len(run) - size + 1))
            if len(run) == 1:
                words.append(run)
        for word in words:
            digest = hashlib.blake2s(word.encode(), digest_size=4).digest()
            vector[int.from_bytes(digest, "little") % self.dimensions] += 1
        return vector.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]


class UnitEmbeddings(Embeddings):
    """Normalize both documents and queries so FAISS inner product is cosine similarity."""

    def __init__(self, wrapped: Embeddings):
        self.wrapped = wrapped

    @staticmethod
    def normalize(vectors: list[list[float]]) -> list[list[float]]:
        try:
            matrix = np.asarray(vectors, dtype=np.float32)
        except (ValueError, TypeError) as exc:
            raise EmbeddingResponseError(INVALID_EMBEDDING_RESPONSE) from exc
        if matrix.ndim != 2 or not matrix.shape[1] or not np.isfinite(matrix).all():
            raise ValueError("Embedding 服务返回了无效向量。")
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return (matrix / np.maximum(norms, 1e-12)).tolist()

    def embed_query(self, text: str) -> list[float]:
        try:
            vector = self.wrapped.embed_query(text)
        except (
            AttributeError,
            KeyError,
            TypeError,
            JSONDecodeError,
            APIResponseValidationError,
        ) as exc:
            raise EmbeddingResponseError(INVALID_EMBEDDING_RESPONSE) from exc
        return self.normalize([vector])[0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        try:
            vectors = self.wrapped.embed_documents(texts)
        except (
            AttributeError,
            KeyError,
            TypeError,
            JSONDecodeError,
            APIResponseValidationError,
        ) as exc:
            raise EmbeddingResponseError(INVALID_EMBEDDING_RESPONSE) from exc
        if not isinstance(vectors, (list, tuple, np.ndarray)):
            raise EmbeddingResponseError(INVALID_EMBEDDING_RESPONSE)
        if len(vectors) != len(texts):
            raise ValueError("Embedding 服务返回的向量数量与输入不一致。")
        return self.normalize(vectors)


def build_embeddings(settings: Settings) -> UnitEmbeddings:
    settings.validate(embeddings=True)
    if settings.mode == "demo":
        return UnitEmbeddings(DemoEmbeddings())
    return UnitEmbeddings(
        OpenAIEmbeddings(
            model=settings.embedding_model,
            api_key=settings.embedding_api_key,
            base_url=settings.embedding_base_url,
            check_embedding_ctx_length=False,  # Send strings, not OpenAI-specific token IDs.
            chunk_size=settings.embedding_batch_size,
            request_timeout=settings.api_timeout,
            max_retries=2,
        )
    )


def build_chat_model(settings: Settings) -> ChatOpenAI:
    settings.validate(chat=True)
    return ChatOpenAI(
        model=settings.llm_model,
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        timeout=settings.api_timeout,
        max_retries=2,
        stream_usage=False,
        use_responses_api=False,
    )
