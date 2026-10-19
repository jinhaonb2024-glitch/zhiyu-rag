from io import BytesIO

import pytest
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from rag.documents import DocumentError, InputFile, clean_text, read_document, split_documents


def make_pdf(pages: list[str | None], password: str | None = None) -> bytes:
    writer = PdfWriter()
    for text in pages:
        page = writer.add_blank_page(width=612, height=792)
        if text is None:
            continue
        font = DictionaryObject(
            {
                NameObject("/Type"): NameObject("/Font"),
                NameObject("/Subtype"): NameObject("/Type1"),
                NameObject("/BaseFont"): NameObject("/Helvetica"),
            }
        )
        page[NameObject("/Resources")] = DictionaryObject(
            {NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})}
        )
        stream = DecodedStreamObject()
        stream.set_data(f"BT /F1 12 Tf 72 700 Td ({text}) Tj ET".encode("ascii"))
        page[NameObject("/Contents")] = stream
    if password:
        writer.encrypt(password)
    output = BytesIO()
    writer.write(output)
    return output.getvalue()


@pytest.mark.parametrize("encoding", ["utf-8-sig", "utf-16", "gb18030"])
def test_chinese_txt_encodings(encoding):
    file = InputFile("../../笔记.txt", "中文学习笔记\r\n\r\nEmbedding 向量检索".encode(encoding))
    docs, warnings = read_document(file)
    assert docs[0].metadata["source"] == "笔记.txt"
    assert docs[0].page_content == "中文学习笔记\n\nEmbedding 向量检索"
    assert not warnings


def test_cleaning_and_split_preserve_provenance():
    assert clean_text("\ufeffＡＢＣ\x00  \r\n\r\n\r\n资料\u200b") == "ABC\n\n资料"
    docs, _ = read_document(InputFile("notes.txt", ("检索与文档分段。" * 70).encode()))
    chunks = split_documents(docs, 100, 20)
    assert len(chunks) > 1
    assert all(len(chunk.page_content) <= 100 for chunk in chunks)
    assert all(chunk.metadata["file_hash"] == docs[0].metadata["file_hash"] for chunk in chunks)
    assert all(chunk.metadata["start_index"] >= 0 for chunk in chunks)
    assert len({chunk.metadata["chunk_id"] for chunk in chunks}) == len(chunks)


def test_pdf_keeps_actual_page_numbers_when_empty_pages_are_skipped():
    docs, warnings = read_document(InputFile("notes.pdf", make_pdf([None, "FAISS retrieval"])))
    assert len(docs) == 1
    assert docs[0].metadata["page"] == 2
    assert "FAISS retrieval" in docs[0].page_content
    assert "1 个无文本页面" in warnings[0]


@pytest.mark.parametrize(
    "file, message",
    [
        (InputFile("empty.txt", b""), "文件为空"),
        (InputFile("bad.pdf", b"not a PDF"), "PDF 解析失败"),
        (InputFile("scan.pdf", make_pdf([None])), "OCR"),
        (InputFile("locked.pdf", make_pdf(["secret"], "password")), "加密"),
        (InputFile("script.py", b"print(1)"), "只支持"),
        (InputFile("binary.txt", b"\x00" * 100), "不像文本"),
    ],
)
def test_invalid_uploads_have_actionable_errors(file, message):
    with pytest.raises(DocumentError, match=message):
        read_document(file)
