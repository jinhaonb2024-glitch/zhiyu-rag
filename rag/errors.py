"""Friendly API errors without echoing provider payloads or credentials."""

import logging
import traceback

from openai import APIConnectionError, APIStatusError, APITimeoutError, AuthenticationError

from rag.documents import DocumentError
from rag.knowledge_base import IndexCompatibilityError, StorageError
from rag.providers import EmbeddingResponseError


def user_error(exc: Exception) -> str:
    if isinstance(exc, AuthenticationError):
        return "API 认证失败，请检查对应服务的 API Key。"
    if isinstance(exc, APITimeoutError):
        return "API 请求超时，请稍后重试，或增大 API_TIMEOUT。"
    if isinstance(exc, APIConnectionError):
        return "无法连接 API，请检查网络及 Base URL。"
    if isinstance(exc, APIStatusError):
        if exc.status_code == 429:
            return "API 请求受限或额度不足，请检查账户额度并稍后重试。"
        return (
            f"API 返回 HTTP {exc.status_code}。请检查模型名、Base URL 和接口兼容性；"
            "Embedding 服务须支持 /embeddings，问答服务须支持 /chat/completions。"
        )
    if isinstance(
        exc, (DocumentError, IndexCompatibilityError, StorageError, EmbeddingResponseError)
    ):
        return str(exc)
    if isinstance(exc, ValueError):
        # Only messages raised by our application are safe to echo.
        if exc.__traceback__:
            tb = exc.__traceback__
            while tb.tb_next:
                tb = tb.tb_next
            if "/rag/" in tb.tb_frame.f_code.co_filename.replace("\\", "/"):
                return str(exc)
    if isinstance(exc, OSError):
        return "本地文件读写失败，请检查磁盘空间、文件权限和知识库目录。"
    # Log call sites only: exception messages and local variables may contain private data.
    logging.getLogger(__name__).error(
        "Unhandled %s\n%s",
        type(exc).__name__,
        "\n".join(
            f"{frame.filename}:{frame.lineno} in {frame.name}"
            for frame in traceback.extract_tb(exc.__traceback__)
        ),
    )
    return f"操作未完成（{type(exc).__name__}）。调用位置已记录到服务日志，请检查配置后重试。"
