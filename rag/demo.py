"""A one-command local pipeline demo: python -m rag.demo"""

import argparse

from rag.config import ROOT, Settings
from rag.documents import InputFile
from rag.knowledge_base import KnowledgeBase
from rag.qa import prepare_answer, stream_answer


def main() -> None:
    parser = argparse.ArgumentParser(description="无需 API Key 的 RAG 流程演示")
    parser.add_argument("--question", default="RAG 的工作流程是什么？")
    args = parser.parse_args()
    settings = Settings()
    kb = KnowledgeBase(settings)
    kb.load()
    report = kb.ingest(
        [
            InputFile(path.name, path.read_bytes())
            for path in sorted((ROOT / "examples").glob("*.txt"))
        ]
    )
    print(
        f"知识库：{kb.stats['files']} 个文档，{kb.stats['chunks']} 个片段；"
        f"本次新增 {report.added_files} 个文档。"
    )
    prepared = prepare_answer(kb, args.question)
    print(f"\n问题：{args.question}\n")
    print("".join(stream_answer(prepared, settings)))
    for i, source in enumerate(prepared.sources, start=1):
        print(f"[{i}] {source.document.metadata['source']} · 相似度 {source.score:.3f}")


if __name__ == "__main__":
    main()
