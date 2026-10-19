"""FAISS retrieval and atomic persistence, without pickle deserialization."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import faiss
import numpy as np
from langchain_core.documents import Document

from rag.config import Settings
from rag.documents import DocumentError, InputFile, read_document, split_documents
from rag.providers import UnitEmbeddings, build_embeddings

MAX_CHUNKS = 10000


class IndexCompatibilityError(ValueError):
    """The persisted index belongs to a different embedding or preprocessing setup."""


class StorageError(ValueError):
    """An unreadable or inconsistent local archive."""


@dataclass(frozen=True)
class SearchHit:
    document: Document
    score: float


@dataclass
class ImportReport:
    added_files: int = 0
    added_chunks: int = 0
    skipped_files: int = 0
    warnings: list[str] = field(default_factory=list)


@dataclass
class VectorIndex:
    index: faiss.IndexFlatIP
    documents: list[Document]


class KnowledgeBase:
    def __init__(self, settings: Settings, path: Path | None = None):
        settings.validate()
        self.settings = settings
        self.path = path if path is not None else settings.index_path
        self.store: VectorIndex | None = None
        self._embeddings: UnitEmbeddings | None = None
        self.updated_at: str | None = None

    @property
    def embeddings(self) -> UnitEmbeddings:
        if self._embeddings is None:
            self._embeddings = build_embeddings(self.settings)
        return self._embeddings

    @property
    def documents(self) -> list[Document]:
        if self.store is None:
            return []
        return self.store.documents.copy()

    @property
    def stats(self) -> dict:
        docs = self.documents
        counts = Counter(doc.metadata["file_hash"] for doc in docs)
        names = {doc.metadata["file_hash"]: doc.metadata["source"] for doc in docs}
        return {
            "files": len(counts),
            "chunks": len(docs),
            "dimensions": self.store.index.d if self.store else 0,
            "sources": [
                {"文件": names[digest], "片段数": count} for digest, count in counts.items()
            ],
        }

    def load(self) -> bool:
        if not self.path.exists():
            return False
        try:
            with np.load(self.path, allow_pickle=False) as archive:
                payload = json.loads(str(archive["metadata"].item()))
                binary = archive["index"].copy()
            if payload["schema"] != 1:
                raise ValueError("Unsupported schema")
            if payload["signature"] != self.settings.index_signature:
                raise IndexCompatibilityError(
                    "Embedding 服务、模型或分段参数与已保存索引不一致。"
                    "请恢复原配置，或重新选择全部文档并点击“重建知识库”。"
                )
            if hashlib.sha256(binary.tobytes()).hexdigest() != payload["index_sha256"]:
                raise ValueError("Index checksum mismatch")
            index = faiss.deserialize_index(binary)
            docs = [
                Document(page_content=row["text"], metadata=row["metadata"])
                for row in payload["documents"]
            ]
            ids = [doc.metadata["chunk_id"] for doc in docs]
            for doc in docs:
                if (
                    not doc.page_content
                    or not {"source", "file_hash", "page"} <= doc.metadata.keys()
                ):
                    raise ValueError("Incomplete document metadata")
            if (
                len(ids) != len(set(ids))
                or index.ntotal != len(docs)
                or index.d != payload["dimensions"]
                or index.metric_type != faiss.METRIC_INNER_PRODUCT
                or not isinstance(index, faiss.IndexFlatIP)
            ):
                raise ValueError("Index metadata mismatch")
            self.store = VectorIndex(index, docs)
            self.updated_at = payload["updated_at"]
            return True
        except IndexCompatibilityError:
            raise
        except (OSError, KeyError, TypeError, ValueError, RuntimeError) as exc:
            raise StorageError(
                "无法加载本地索引，请检查配置及文件权限，或重新选择文档并重建知识库。"
            ) from exc

    def _save(self, store: VectorIndex) -> str:
        updated_at = datetime.now(UTC).isoformat()
        binary = faiss.serialize_index(store.index)
        docs = store.documents
        payload = {
            "schema": 1,
            "signature": self.settings.index_signature,
            "dimensions": store.index.d,
            "updated_at": updated_at,
            "index_sha256": hashlib.sha256(binary.tobytes()).hexdigest(),
            "documents": [{"text": doc.page_content, "metadata": doc.metadata} for doc in docs],
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=self.path.parent, suffix=".tmp", delete=False
            ) as f:
                temp_path = Path(f.name)
                np.savez_compressed(
                    f, index=binary, metadata=np.array(json.dumps(payload, ensure_ascii=False))
                )
                f.flush()
                os.fsync(f.fileno())
            os.replace(temp_path, self.path)
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)
        return updated_at

    def ingest(self, files: list[InputFile], *, replace: bool = False) -> ImportReport:
        self.settings.validate(embeddings=True)
        if not files:
            raise DocumentError("请先选择要导入的 PDF 或 TXT 文件。")
        if len(files) > 20:
            raise DocumentError("每次最多导入 20 个文件，请分批导入。")
        known = set() if replace else {doc.metadata["file_hash"] for doc in self.documents}
        chunks: list[Document] = []
        report = ImportReport()
        for file in files:
            if file.digest in known:
                report.skipped_files += 1
                continue
            try:
                docs, warnings = read_document(file)
                new_chunks = split_documents(
                    docs, self.settings.chunk_size, self.settings.chunk_overlap
                )
            except DocumentError as exc:
                report.warnings.append(str(exc))
                continue
            known.add(file.digest)
            chunks.extend(new_chunks)
            report.warnings.extend(warnings)
            report.added_files += 1
        report.added_chunks = len(chunks)
        if not chunks:
            if report.skipped_files and not replace:
                return report
            raise DocumentError("没有可导入的新内容。" + "\n" + "\n".join(report.warnings))
        old_count = 0 if replace or self.store is None else self.store.index.ntotal
        if old_count + len(chunks) > MAX_CHUNKS:
            raise DocumentError("知识库最多支持 10000 个片段，请减少文档或增大分段长度。")

        # Build a candidate first. API or disk failures never mutate the active knowledge base.
        vectors = self.embeddings.embed_documents([doc.page_content for doc in chunks])
        if replace or self.store is None:
            candidate = VectorIndex(faiss.IndexFlatIP(len(vectors[0])), [])
        else:
            if len(vectors[0]) != self.store.index.d:
                raise IndexCompatibilityError(
                    "Embedding 向量维度发生变化，请重新导入文档重建索引。"
                )
            candidate = VectorIndex(faiss.clone_index(self.store.index), self.documents)
        candidate.index.add(np.asarray(vectors, dtype=np.float32))
        candidate.documents.extend(chunks)
        updated_at = self._save(candidate)
        self.store = candidate
        self.updated_at = updated_at
        return report

    def search(
        self, question: str, *, top_k: int | None = None, min_similarity: float | None = None
    ) -> list[SearchHit]:
        question = question.strip()
        if not question:
            raise ValueError("问题不能为空。")
        if len(question) > 2000:
            raise ValueError("问题请控制在 2000 字符以内。")
        if self.store is None:
            return []
        k = self.settings.top_k if top_k is None else top_k
        threshold = self.settings.min_similarity if min_similarity is None else min_similarity
        if not 1 <= k <= 20 or not -1 <= threshold <= 1:
            raise ValueError("检索参数超出有效范围。")
        vector = self.embeddings.embed_query(question)
        if len(vector) != self.store.index.d:
            raise IndexCompatibilityError("查询向量与索引维度不同，请检查 Embedding 模型并重建。")
        scores, positions = self.store.index.search(
            np.asarray([vector], dtype=np.float32), min(k, self.store.index.ntotal)
        )
        return [
            SearchHit(self.store.documents[int(position)], float(np.clip(score, -1, 1)))
            for score, position in zip(scores[0], positions[0], strict=True)
            if position >= 0 and score >= threshold
        ]
