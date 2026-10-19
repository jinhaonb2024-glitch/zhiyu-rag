from dataclasses import replace

import numpy as np
import pytest

from rag.config import Settings
from rag.documents import DocumentError, InputFile
from rag.knowledge_base import IndexCompatibilityError, KnowledgeBase, StorageError
from rag.qa import NO_EVIDENCE, prepare_answer, stream_answer


@pytest.fixture
def kb(tmp_path):
    return KnowledgeBase(Settings(), tmp_path / "中文目录" / "knowledge.npz")


def notes():
    return [
        InputFile("FAISS.txt", "FAISS 使用归一化向量与内积实现余弦相似度检索。".encode()),
        InputFile("生活.txt", "番茄炒蛋需要鸡蛋、番茄和盐，先炒鸡蛋再加入番茄。".encode()),
    ]


def test_retrieval_deduplication_and_unicode_path_reload(kb):
    report = kb.ingest(notes())
    assert (report.added_files, report.added_chunks) == (2, 2)
    hits = kb.search("FAISS 余弦相似度检索", min_similarity=0.05)
    assert hits[0].document.metadata["source"] == "FAISS.txt"
    assert 0 < hits[0].score <= 1
    assert kb.ingest(notes()).skipped_files == 2
    restored = KnowledgeBase(kb.settings, kb.path)
    assert restored.load()
    assert restored.stats == kb.stats
    assert restored.search("FAISS 余弦相似度检索")[0].score == pytest.approx(hits[0].score)
    assert not list(kb.path.parent.glob("*.tmp"))


def test_incremental_import_and_explicit_replace(kb):
    kb.ingest(notes()[:1])
    kb.ingest(notes()[1:])
    assert kb.stats["files"] == 2
    kb.ingest(notes()[:1], replace=True)
    assert kb.stats["files"] == 1
    restored = KnowledgeBase(kb.settings, kb.path)
    restored.load()
    assert restored.stats["files"] == 1


def test_failed_save_does_not_change_live_or_persisted_index(kb, monkeypatch):
    kb.ingest(notes()[:1])
    original = kb.path.read_bytes()

    def fail_save(*args):
        raise OSError("disk full")

    monkeypatch.setattr(kb, "_save", fail_save)
    with pytest.raises(OSError):
        kb.ingest(notes()[1:])
    assert kb.stats["files"] == 1
    assert kb.path.read_bytes() == original


def test_failed_rebuild_preserves_old_data(kb):
    kb.ingest(notes())
    with pytest.raises(DocumentError):
        kb.ingest([InputFile("empty.txt", b"")], replace=True)
    assert kb.stats["files"] == 2


def test_bad_files_are_reported_while_valid_files_are_imported(kb):
    report = kb.ingest([notes()[0], InputFile("bad.pdf", b"invalid")])
    assert report.added_files == 1
    assert "PDF 解析失败" in report.warnings[0]


def test_incompatible_index_is_rejected_before_query(kb):
    kb.ingest(notes())
    for settings in [
        replace(kb.settings, chunk_size=800),
        replace(kb.settings, mode="api", embedding_model="other-model"),
    ]:
        with pytest.raises(IndexCompatibilityError):
            KnowledgeBase(settings, kb.path).load()


def test_corrupted_archive_rejected(kb):
    kb.path.parent.mkdir(parents=True)
    kb.path.write_bytes(b"not an archive")
    with pytest.raises(StorageError):
        kb.load()


def test_zero_query_and_out_of_scope_question_do_not_invent_answer(kb):
    kb.ingest(notes()[:1])
    assert kb.search("!!!", min_similarity=0.01) == []
    strict_kb = KnowledgeBase(replace(kb.settings, min_similarity=0.99), kb.path)
    strict_kb.load()
    prepared = prepare_answer(strict_kb, "明天火星上的气温是多少？")
    assert not prepared.sources
    assert "".join(stream_answer(prepared, strict_kb.settings)) == NO_EVIDENCE
    with pytest.raises(ValueError, match="不能为空"):
        kb.search(" ")


def test_context_budget_limits_and_displayed_sources_match_prompt(kb):
    kb.ingest([InputFile("long.txt", ("FAISS 向量检索相关资料。" * 100).encode())])
    limited = KnowledgeBase(replace(kb.settings, max_context_chars=120), kb.path)
    limited.load()
    prepared = prepare_answer(limited, "FAISS 向量检索")
    assert len(prepared.context) <= 120
    assert prepared.sources
    assert all(hit.document.page_content in prepared.context for hit in prepared.sources)
    assert "未调用大模型" in "".join(stream_answer(prepared, limited.settings))


def test_saved_archive_contains_no_api_keys(kb):
    kb.ingest(notes())
    with np.load(kb.path, allow_pickle=False) as archive:
        metadata = str(archive["metadata"].item())
        assert "api_key" not in metadata
        assert "FAISS.txt" in metadata
