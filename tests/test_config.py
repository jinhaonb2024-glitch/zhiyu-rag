from dataclasses import replace

import pytest

from rag.config import Settings


def test_console_url_is_rejected_on_apply_without_sending_request():
    settings = Settings(mode="api", embedding_base_url="https://cloud.siliconflow.cn/me/models")
    with pytest.raises(ValueError, match="https://api.siliconflow.cn/v1"):
        settings.validate()


@pytest.mark.parametrize(
    "url",
    [
        "https://embedding.test/v1/embeddings",
        "https://chat.test/v1/chat/completions/",
        "https://api.deepseek.com/models",
        "https://platform.deepseek.com/api_keys",
        "https://embedding.test/v1?api_key=secret",
    ],
)
def test_webpages_and_full_resource_urls_rejected(url):
    with pytest.raises(ValueError):
        Settings(mode="api", embedding_base_url=url).validate()


def test_api_prefixes_allowed_without_requiring_keys_on_apply():
    settings = Settings(
        mode="api",
        llm_base_url="https://api.deepseek.com",
        embedding_base_url="https://api.siliconflow.cn/v1",
    )
    settings.validate()
    replace(settings, embedding_base_url="http://localhost:8000/custom/v1").validate()
    with pytest.raises(ValueError, match="API Key"):
        settings.validate(embeddings=True)
