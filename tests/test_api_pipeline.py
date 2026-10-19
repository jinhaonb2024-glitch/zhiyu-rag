"""Exercise real LangChain/OpenAI clients with a simulated HTTP transport, no paid API."""

import json
from dataclasses import replace

import httpx
import numpy as np
import pytest
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from openai import AuthenticationError

from rag import providers
from rag.config import Settings
from rag.documents import InputFile
from rag.errors import user_error
from rag.knowledge_base import KnowledgeBase
from rag.providers import DemoEmbeddings, EmbeddingResponseError, UnitEmbeddings
from rag.qa import prepare_answer, stream_answer


def test_real_api_adapters_embed_retrieve_and_stream(tmp_path, monkeypatch):
    requests = []

    def handle(request):
        body = json.loads(request.content)
        requests.append((request, body))
        if request.url.path == "/v1/embeddings":
            assert request.url.host == "embedding.test"
            assert request.headers["authorization"] == "Bearer test-embedding-key"
            assert body["model"] == "test-embedding"
            assert all(isinstance(text, str) for text in body["input"])
            vectors = DemoEmbeddings().embed_documents(body["input"])
            return httpx.Response(
                200,
                json={
                    "object": "list",
                    "model": body["model"],
                    "data": [
                        {"object": "embedding", "index": i, "embedding": vector}
                        for i, vector in enumerate(vectors)
                    ],
                    "usage": {"prompt_tokens": 10, "total_tokens": 10},
                },
            )
        assert request.url.host == "chat.test"
        assert request.url.path == "/v1/chat/completions"
        assert request.headers["authorization"] == "Bearer test-chat-key"
        assert body["stream"] is True
        assert body["model"] == "test-chat"
        assert "余弦相似度" in body["messages"][-1]["content"]
        assert "[1]" in body["messages"][-1]["content"]
        assert "资料不足" in body["messages"][0]["content"]
        events = []
        for text in ["FAISS 使用归一化向量", "实现余弦相似度检索。[1]"]:
            events.append(
                "data: "
                + json.dumps(
                    {
                        "id": "chatcmpl-test",
                        "object": "chat.completion.chunk",
                        "created": 0,
                        "model": "test-chat",
                        "choices": [
                            {"index": 0, "delta": {"content": text}, "finish_reason": None}
                        ],
                    },
                    ensure_ascii=False,
                )
                + "\n\n"
            )
        return httpx.Response(
            200,
            content=("".join(events) + "data: [DONE]\n\n").encode(),
            headers={"content-type": "text/event-stream"},
        )

    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        monkeypatch.setattr(
            providers,
            "OpenAIEmbeddings",
            lambda **kwargs: OpenAIEmbeddings(http_client=client, **kwargs),
        )
        monkeypatch.setattr(
            providers, "ChatOpenAI", lambda **kwargs: ChatOpenAI(http_client=client, **kwargs)
        )
        settings = Settings(
            mode="api",
            embedding_api_key="test-embedding-key",
            embedding_base_url="https://embedding.test/v1",
            embedding_model="test-embedding",
            llm_api_key="test-chat-key",
            llm_base_url="https://chat.test/v1",
            llm_model="test-chat",
        )
        kb = KnowledgeBase(settings, tmp_path / "api.npz")
        kb.ingest([InputFile("FAISS.txt", "FAISS 使用归一化向量实现余弦相似度检索。".encode())])
        prepared = prepare_answer(kb, "FAISS 如何实现余弦相似度检索？")
        result = "".join(stream_answer(prepared, settings))
        assert result == "FAISS 使用归一化向量实现余弦相似度检索。[1]"
        assert len(requests) == 3  # document embeddings, query embedding, chat stream
        assert prepared.sources[0].document.metadata["source"] == "FAISS.txt"
        restored = KnowledgeBase(settings, kb.path)
        assert restored.load()
        assert restored.search("FAISS 余弦相似度检索")
        assert len(requests) == 4  # loading the index does not re-embed stored documents
        with np.load(kb.path, allow_pickle=False) as archive:
            assert "test-embedding-key" not in str(archive["metadata"].item())
        with pytest.raises(ValueError, match="API Key"):
            prepare_answer(KnowledgeBase(replace(settings, llm_api_key=""), kb.path), "FAISS")


def test_normalization_and_invalid_provider_vectors():
    assert UnitEmbeddings.normalize([[3, 4]])[0] == pytest.approx([0.6, 0.8])
    assert UnitEmbeddings.normalize([[0, 0]]) == [[0, 0]]
    with pytest.raises(ValueError, match="无效向量"):
        UnitEmbeddings.normalize([[float("nan"), 1]])


def test_provider_errors_do_not_echo_sensitive_response():
    response = httpx.Response(401, request=httpx.Request("POST", "https://chat.test/v1"))
    error = AuthenticationError("secret-provider-response", response=response, body=None)
    assert "认证失败" in user_error(error)
    assert "secret" not in user_error(error)


@pytest.mark.parametrize("method", ["embed_documents", "embed_query"])
def test_html_embedding_response_is_actionable_and_does_not_echo_body(method, monkeypatch):
    with httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200, text="<html>secret-provider-page</html>", headers={"content-type": "text/html"}
            )
        )
    ) as client:
        monkeypatch.setattr(
            providers,
            "OpenAIEmbeddings",
            lambda **kwargs: OpenAIEmbeddings(http_client=client, **kwargs),
        )
        embeddings = providers.build_embeddings(Settings(mode="api", embedding_api_key="test"))
        with pytest.raises(EmbeddingResponseError) as caught:
            getattr(embeddings, method)(["test"] if method == "embed_documents" else "test")
        message = user_error(caught.value)
        assert "Base URL" in message
        assert "secret-provider-page" not in message


def test_unknown_error_logs_stack_but_never_message(caplog):
    try:
        raise RuntimeError("secret-api-key-and-document-content")
    except RuntimeError as exc:
        message = user_error(exc)
    assert "RuntimeError" in message
    assert "Unhandled RuntimeError" in caplog.text
    assert "secret-api-key-and-document-content" not in caplog.text
