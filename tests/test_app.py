import json

import httpx
import pytest
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from streamlit.testing.v1 import AppTest

from rag import providers
from rag.config import ROOT, Settings
from rag.documents import InputFile
from rag.knowledge_base import KnowledgeBase
from rag.providers import DemoEmbeddings


def test_streamlit_demo_import_question_sources_and_reload(tmp_path, monkeypatch):
    monkeypatch.setattr(Settings, "from_env", classmethod(lambda cls: Settings()))
    monkeypatch.setattr(Settings, "index_path", property(lambda self: tmp_path / "demo.npz"))
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
    assert not app.exception
    assert app.chat_input[0].disabled
    next(button for button in app.button if button.label == "加载示例资料").click().run()
    assert not app.exception
    assert not app.chat_input[0].disabled
    assert app.metric[0].value == "3"
    app.chat_input[0].set_value("FAISS 如何实现余弦相似度检索？").run()
    assert not app.exception
    assert len(app.chat_message) == 2
    assert "未调用大模型" in app.chat_message[1].markdown[0].value
    assert any("相似度" in expander.label for expander in app.expander)
    assert app.session_state["messages"][1]["sources"]
    restarted = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
    assert not restarted.exception
    assert restarted.metric[0].value == "3"
    assert not restarted.chat_input[0].disabled


@pytest.mark.parametrize("interrupted", [False, True], ids=["success", "interrupted-stream"])
def test_api_page_import_answer_sources_and_restart(tmp_path, monkeypatch, interrupted):
    requests = []

    class AnswerStream(httpx.SyncByteStream):
        def __iter__(self):
            for text in ["FAISS 使用归一化向量", "实现余弦相似度检索。[1]"]:
                event = {
                    "id": "chatcmpl-ui-test",
                    "object": "chat.completion.chunk",
                    "created": 0,
                    "model": "test-chat",
                    "choices": [{"index": 0, "delta": {"content": text}}],
                }
                yield ("data: " + json.dumps(event) + "\n\n").encode()
                if interrupted:
                    raise httpx.ReadError("secret-provider-response")
            yield b"data: [DONE]\n\n"

    def handle(request):
        body = json.loads(request.content)
        requests.append((request.url.path, body))
        if request.url.path == "/v1/embeddings":
            assert request.url.host == "embedding.test"
            assert request.headers["authorization"] == "Bearer test-embedding-key"
            vectors = DemoEmbeddings().embed_documents(body["input"])
            return httpx.Response(
                200,
                json={
                    "data": [
                        {"index": i, "object": "embedding", "embedding": vector}
                        for i, vector in enumerate(vectors)
                    ],
                    "model": body["model"],
                    "object": "list",
                    "usage": {"prompt_tokens": 10, "total_tokens": 10},
                },
            )
        assert request.url.host == "chat.test"
        assert request.url.path == "/v1/chat/completions"
        assert request.headers["authorization"] == "Bearer test-chat-key"
        assert body["stream"] is True
        assert "FAISS与Embedding.txt" in body["messages"][-1]["content"]
        return httpx.Response(
            200, stream=AnswerStream(), headers={"content-type": "text/event-stream"}
        )

    settings = Settings(
        mode="api",
        llm_api_key="test-chat-key",
        llm_base_url="https://chat.test/v1",
        llm_model="test-chat",
        embedding_api_key="test-embedding-key",
        embedding_base_url="https://embedding.test/v1",
        embedding_model="test-embedding",
    )
    monkeypatch.setattr(Settings, "from_env", classmethod(lambda cls: settings))
    monkeypatch.setattr(Settings, "index_path", property(lambda self: tmp_path / "api.npz"))
    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        monkeypatch.setattr(
            providers,
            "OpenAIEmbeddings",
            lambda **kwargs: OpenAIEmbeddings(http_client=client, **kwargs),
        )
        monkeypatch.setattr(
            providers, "ChatOpenAI", lambda **kwargs: ChatOpenAI(http_client=client, **kwargs)
        )
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
        assert not app.exception
        next(button for button in app.button if button.label == "加载示例资料").click().run()
        assert not app.exception
        assert app.metric[0].value == "3"
        assert app.metric[2].value == "API 问答"
        app.chat_input[0].set_value("FAISS 如何实现余弦相似度检索？").run()
        assert not app.exception
        answer = app.session_state["messages"][1]
        assert answer["sources"][0].document.metadata["source"] == "FAISS与Embedding.txt"
        assert answer["mode"] == "test-chat"
        assert "离线演示" not in answer["content"]
        if interrupted:
            assert answer["content"] == "FAISS 使用归一化向量"
            assert answer["error"]
            assert any("未完成回答" in warning.value for warning in app.warning)
            assert "secret-provider-response" not in answer["error"]
        else:
            assert answer["content"] == "FAISS 使用归一化向量实现余弦相似度检索。[1]"
            assert not answer["error"]
        assert any("相似度" in expander.label for expander in app.expander)
        assert app.get("download_button")
        request_count = len(requests)
        restarted = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
        assert not restarted.exception
        assert restarted.metric[0].value == "3"
        assert not restarted.chat_input[0].disabled
        assert len(requests) == request_count  # Restart restores the index without API calls.


def test_same_filename_versions_have_distinct_previews(tmp_path, monkeypatch):
    index_path = tmp_path / "demo.npz"
    monkeypatch.setattr(Settings, "from_env", classmethod(lambda cls: Settings()))
    monkeypatch.setattr(Settings, "index_path", property(lambda self: index_path))
    files = [
        InputFile("笔记.txt", "第一版资料:介绍 FAISS 检索。".encode()),
        InputFile("笔记.txt", "第二版资料:补充 Embedding 模型配置。".encode()),
    ]
    kb = KnowledgeBase(Settings())
    kb.ingest(files)
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
    assert not app.exception
    assert app.metric[0].value == "2"
    previews = [expander for expander in app.expander if expander.label.startswith("预览 ·")]
    assert len(previews) == 2
    assert previews[0].label != previews[1].label
    assert [preview.text[0].value for preview in previews] == [
        file.content.decode() for file in files
    ]
