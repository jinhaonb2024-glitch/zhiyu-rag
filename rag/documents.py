"""Read PDF/TXT uploads without using a user-supplied filename as a disk path."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass
from io import BytesIO
from pathlib import PurePosixPath

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

MAX_FILE_BYTES = 20 * 1024 * 1024
MAX_PDF_PAGES = 500


class DocumentError(ValueError):
    """An actionable upload/parsing error safe to show in the UI."""


@dataclass(frozen=True)
class InputFile:
    name: str
    content: bytes

    @property
    def safe_name(self) -> str:
        name = PurePosixPath(self.name.replace("\\", "/")).name
        return "".join(c for c in name if c.isprintable())[:200] or "document.txt"

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.content).hexdigest()


def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\x00", "").replace("\ufeff", "").replace("\u200b", "")
    text = "\n".join(re.sub(r"[^\S\n]+", " ", line).strip() for line in text.split("\n"))
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def read_document(file: InputFile) -> tuple[list[Document], list[str]]:
    name = file.safe_name
    if not file.content:
        raise DocumentError(f"{name}：文件为空。")
    if len(file.content) > MAX_FILE_BYTES:
        raise DocumentError(f"{name}：文件超过 20 MB 限制。")
    suffix = PurePosixPath(name).suffix.lower()
    metadata = {"source": name, "file_hash": file.digest}
    warnings: list[str] = []
    if suffix == ".txt":
        encodings = (
            ("utf-16",)
            if file.content.startswith((b"\xff\xfe", b"\xfe\xff"))
            else ("utf-8-sig", "gb18030")
        )
        for encoding in encodings:
            try:
                text = file.content.decode(encoding)
                break
            except UnicodeError:
                continue
        else:
            raise DocumentError(f"{name}：无法解码，请将 TXT 另存为 UTF-8。")
        # Reject binary data that happens to be decodable by a permissive encoding.
        controls = sum(ord(c) < 32 and c not in "\n\r\t" for c in text)
        if controls / max(1, len(text)) > 0.02:
            raise DocumentError(f"{name}：内容不像文本文件，请另存为 UTF-8 TXT。")
        docs = [Document(page_content=clean_text(text), metadata={**metadata, "page": None})]
    elif suffix == ".pdf":
        try:
            reader = PdfReader(BytesIO(file.content))
            if reader.is_encrypted and not reader.decrypt(""):
                raise DocumentError(f"{name}：PDF 已加密，请先解密再导入。")
            if len(reader.pages) > MAX_PDF_PAGES:
                raise DocumentError(f"{name}：PDF 超过 500 页，请拆分后导入。")
            docs = [
                Document(
                    page_content=clean_text(page.extract_text() or ""),
                    metadata={**metadata, "page": number},
                )
                for number, page in enumerate(reader.pages, start=1)
            ]
        except DocumentError:
            raise
        except Exception as exc:
            raise DocumentError(f"{name}：PDF 解析失败，请检查文件是否损坏。") from exc
        empty = sum(not doc.page_content for doc in docs)
        if empty:
            warnings.append(f"{name}：跳过 {empty} 个无文本页面；扫描页需先做 OCR。")
    else:
        raise DocumentError(f"{name}：只支持 PDF 和 TXT 文件。")
    docs = [doc for doc in docs if doc.page_content]
    if not docs:
        raise DocumentError(f"{name}：没有可提取的文本。扫描 PDF 请先做 OCR。")
    return docs, warnings


def split_documents(docs: list[Document], chunk_size: int, chunk_overlap: int) -> list[Document]:
    if chunk_size < 1 or not 0 <= chunk_overlap < chunk_size:
        raise ValueError("分段长度和重叠长度不正确。")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
        separators=["\n\n", "\n", "。", "！", "？", ";", "；", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    for number, chunk in enumerate(chunks, start=1):
        chunk.metadata["chunk_number"] = number
        identity = (
            f"{chunk.metadata['file_hash']}:{chunk.metadata['page']}:"
            f"{chunk.metadata['start_index']}:{chunk.page_content}"
        )
        chunk.metadata["chunk_id"] = hashlib.sha256(identity.encode()).hexdigest()
    return chunks
