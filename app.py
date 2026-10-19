"""Run with: python -m streamlit run app.py"""

from __future__ import annotations

import time
from dataclasses import replace

import streamlit as st

from rag.config import ROOT, Settings
from rag.documents import InputFile
from rag.errors import user_error
from rag.knowledge_base import KnowledgeBase, SearchHit
from rag.qa import prepare_answer, stream_answer

st.set_page_config(page_title="知屿 · 个人知识库", page_icon="📚", layout="wide")
st.markdown(
    """<style>
.block-container {max-width: 1200px; padding-top: 2rem;}
[data-testid="stSidebar"] {border-right: 1px solid #E5EAF1;}
.eyebrow {color: #0F766E; font-size: .78rem; letter-spacing: .16em; font-weight: 700;}
.intro {color: #637084; margin-bottom: 1.5rem; line-height: 1.8;}
[data-testid="stMetric"] {background: white; padding: 1rem 1.2rem;
border: 1px solid #E5EAF1; border-radius: 12px;}
[data-testid="stChatMessage"] {border: 1px solid #E5EAF1; border-radius: 12px;}
</style>""",
    unsafe_allow_html=True,
)


def render_sources(sources: list[SearchHit]) -> None:
    if not sources:
        return
    st.caption(f"本次使用 {len(sources)} 个参考片段 · 余弦相似度越高表示向量越接近，并非答案置信度")
    for number, hit in enumerate(sources, start=1):
        meta = hit.document.metadata
        page = f" · 第 {meta['page']} 页" if meta.get("page") else ""
        label = f"[{number}] {meta['source']}{page} · 相似度 {hit.score:.3f}"
        with st.expander(label):
            st.text(hit.document.page_content)
            st.caption(f"片段 {meta['chunk_number']} · ID {meta['chunk_id'][:12]}")


def render_message(message: dict) -> None:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("error"):
            st.warning(message["error"])
        if message.get("sources"):
            render_sources(message["sources"])
        if "seconds" in message:
            st.caption(f"{message['mode']} · 用时 {message['seconds']:.2f} 秒")


if "settings" not in st.session_state:
    try:
        initial_settings = Settings.from_env()
        initial_settings.validate()
        st.session_state.settings = initial_settings
    except (ValueError, OSError):
        st.error(".env 配置无效，请对照 .env.example 检查模式及数值参数。")
        st.stop()

settings: Settings = st.session_state.settings
with st.sidebar:
    st.markdown("### 📚 知屿")
    st.caption("让分散的知识，成为随时可查的答案。")
    with st.expander("⚙️ 模型与检索设置", expanded=False), st.form("model_settings"):
        mode = st.selectbox(
            "运行模式",
            ["demo", "api"],
            index=0 if settings.mode == "demo" else 1,
            format_func=lambda value: "离线演示" if value == "demo" else "API 问答",
        )
        st.caption("问答模型 · OpenAI 兼容 Chat Completions 接口")
        llm_url = st.text_input("问答 Base URL", value=settings.llm_base_url)
        llm_model = st.text_input("问答模型名", value=settings.llm_model)
        llm_key = st.text_input("问答 API Key", value=settings.llm_api_key, type="password")
        st.caption("Embedding 服务 · 支持 /embeddings 接口")
        emb_url = st.text_input(
            "Embedding Base URL",
            value=settings.embedding_base_url,
            help="填写 API 接口前缀。硅基流动：https://api.siliconflow.cn/v1；不要填模型广场网页。",
        )
        emb_model = st.text_input("Embedding 模型名", value=settings.embedding_model)
        emb_key = st.text_input(
            "Embedding API Key", value=settings.embedding_api_key, type="password"
        )
        chunk_size = st.number_input("分段长度（字符）", 100, 4000, settings.chunk_size, step=100)
        overlap = st.number_input("重叠长度（字符）", 0, 3999, settings.chunk_overlap, step=20)
        top_k = st.slider("检索片段数量", 1, 20, settings.top_k)
        threshold = st.slider("最低余弦相似度", -1.0, 1.0, settings.min_similarity, step=0.01)
        st.caption("切换 Embedding 或分段参数后需重建索引。密钥仅用于当前会话，不写入索引。")
        if st.form_submit_button("应用设置", width="stretch"):
            candidate = replace(
                settings,
                mode=mode,
                llm_base_url=llm_url.strip(),
                llm_model=llm_model.strip(),
                llm_api_key=llm_key.strip(),
                embedding_base_url=emb_url.strip(),
                embedding_model=emb_model.strip(),
                embedding_api_key=emb_key.strip(),
                chunk_size=int(chunk_size),
                chunk_overlap=int(overlap),
                top_k=top_k,
                min_similarity=threshold,
            )
            try:
                candidate.validate()
                st.session_state.settings = candidate
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))

# Session-owned resources: API keys and active documents are never globally cached.
if st.session_state.get("loaded_settings") != settings:
    kb = KnowledgeBase(settings)
    issue = ""
    try:
        kb.load()
    except Exception as exc:
        issue = user_error(exc)
    st.session_state.kb = kb
    st.session_state.loaded_settings = settings
    st.session_state.index_issue = issue
    st.session_state.messages = []
    st.session_state.pop("import_report", None)

kb: KnowledgeBase = st.session_state.kb
index_issue = st.session_state.index_issue


def import_files(files: list[InputFile], *, rebuild: bool = False) -> None:
    try:
        with st.spinner("正在读取文档、分段并建立向量索引…"):
            report = kb.ingest(files, replace=rebuild)
        st.session_state.import_report = report
        st.session_state.index_issue = ""
        if report.added_files:
            st.session_state.messages = []
        st.rerun()
    except Exception as exc:
        st.error(user_error(exc))


with st.sidebar:
    st.divider()
    st.markdown("#### 导入你的资料")
    uploads = st.file_uploader(
        "选择 PDF / TXT",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        help="单个文件最多 20 MB，每次最多 20 个。扫描 PDF 需先做 OCR。",
    )
    if st.button("导入并建立索引", type="primary", width="stretch", disabled=bool(index_issue)):
        import_files([InputFile(file.name, file.getvalue()) for file in uploads])
    if st.button("加载示例资料", width="stretch", disabled=bool(index_issue)):
        import_files(
            [
                InputFile(path.name, path.read_bytes())
                for path in sorted((ROOT / "examples").glob("*.txt"))
            ]
        )
    with st.expander("重建知识库"):
        st.caption("用当前选择的文件替换整个知识库。请先选齐全部资料；成功保存后覆盖原索引。")
        if st.button("重建知识库", width="stretch", disabled=not uploads):
            import_files([InputFile(file.name, file.getvalue()) for file in uploads], rebuild=True)
    if settings.mode == "api":
        st.caption("API 模式会将文档片段发送给 Embedding 服务，问题和参考片段发送给问答服务。")
    else:
        st.caption("离线演示：本地词面向量检索 + 原文摘录，不调用外部模型。")
    st.caption("本机个人使用 · PDF / TXT · FAISS")

st.markdown('<div class="eyebrow">PERSONAL KNOWLEDGE WORKSPACE</div>', unsafe_allow_html=True)
st.title("你的知识，随问随用")
st.markdown(
    '<div class="intro">导入学习笔记与技术文档，用自然语言查找答案。'
    "每次回答，都能回到资料本身。</div>",
    unsafe_allow_html=True,
)
stats = kb.stats
col1, col2, col3 = st.columns(3)
col1.metric("已导入文档", stats["files"])
col2.metric("可检索片段", stats["chunks"])
col3.metric("当前模式", "离线演示" if settings.mode == "demo" else "API 问答")

if index_issue:
    st.warning(index_issue)
if report := st.session_state.get("import_report"):
    st.success(
        f"新增 {report.added_files} 个文档、{report.added_chunks} 个片段；"
        f"跳过 {report.skipped_files} 个重复文档。"
    )
    for warning in report.warnings:
        st.warning(warning)

chat_tab, library_tab, help_tab = st.tabs(["💬 知识问答", "📂 我的文档", "📖 使用说明"])
with chat_tab:
    st.caption("每次问题独立检索；追问时请写明主题。修改知识库或设置后会清空当前问答记录。")
    if not stats["chunks"]:
        st.info("从左侧导入文档，或点击“加载示例资料”开始体验。")
    elif settings.mode == "demo":
        st.info("当前为离线演示，只展示检索到的原文。使用大模型回答请在左侧切换到 API 问答。")

    pending = None
    if stats["chunks"] and not st.session_state.messages:
        st.markdown("**可以从这些问题开始**")
        suggestions = [
            "RAG 的工作流程是什么？",
            "为什么文本分段要设置重叠？",
            "FAISS 如何实现余弦相似度检索？",
        ]
        for column, question in zip(st.columns(3), suggestions, strict=True):
            if column.button(question, width="stretch", disabled=bool(index_issue)):
                pending = question

    for message in st.session_state.messages:
        render_message(message)

    question = st.chat_input(
        "向知识库提问，例如：RAG 的工作流程是什么？",
        max_chars=2000,
        disabled=not stats["chunks"] or bool(index_issue),
    )
    if question or pending:
        question = question or pending
        started = time.perf_counter()
        user_message = {"role": "user", "content": question}
        st.session_state.messages.append(user_message)
        render_message(user_message)
        with st.chat_message("assistant"):
            sources = []
            pieces = []
            error = ""
            placeholder = st.empty()
            try:
                with st.spinner("正在检索参考资料…"):
                    prepared = prepare_answer(kb, question)
                sources = prepared.sources
                for piece in stream_answer(prepared, settings):
                    pieces.append(piece)
                    placeholder.markdown("".join(pieces) + " ▌")
            except Exception as exc:
                error = user_error(exc)
            content = "".join(pieces)
            if error:
                placeholder.markdown(content or "本次回答未完成。")
                st.warning(error + (" 已显示内容为未完成回答。" if content else ""))
            else:
                placeholder.markdown(content)
            render_sources(sources)
            elapsed = time.perf_counter() - started
            mode_label = "离线摘录" if settings.mode == "demo" else settings.llm_model
            st.caption(f"{mode_label} · 用时 {elapsed:.2f} 秒")
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": content or "本次回答未完成。",
                "sources": sources,
                "seconds": elapsed,
                "mode": mode_label,
                "error": error,
            }
        )
    if st.session_state.messages:
        left, right = st.columns([1, 3])
        if left.button("清空问答记录"):
            st.session_state.messages = []
            st.rerun()
        transcript = "\n\n".join(
            f"{'问题' if msg['role'] == 'user' else '回答'}：\n{msg['content']}"
            + (f"\n\n错误：{msg['error']}" if msg.get("error") else "")
            + "".join(
                f"\n\n[{i}] {hit.document.metadata['source']} · "
                f"页码 {hit.document.metadata.get('page') or '—'}\n{hit.document.page_content}"
                for i, hit in enumerate(msg.get("sources", []), start=1)
            )
            for msg in st.session_state.messages
        )
        right.download_button(
            "导出本次问答", transcript, file_name="knowledge-qa.md", mime="text/markdown"
        )

with library_tab:
    if stats["sources"]:
        st.dataframe(stats["sources"], hide_index=True, width="stretch")
        st.caption(f"向量维度：{stats['dimensions']} · 知识库自动保存在本机，重启后恢复。")
        st.caption(
            "同内容文件只导入一次；同名文件内容改变会作为新文档导入。删除旧版本请重建知识库。"
        )
        previews = {}
        for document in kb.documents:
            previews.setdefault(document.metadata["file_hash"], document)
        for digest, preview in previews.items():
            with st.expander(f"预览 · {preview.metadata['source']} · {digest[:8]}"):
                st.text(preview.page_content)
    else:
        st.info("这里会展示已导入的文件及片段数量。")

with help_tab:
    st.markdown("""
1. **导入资料**：上传 PDF / TXT，点击“导入并建立索引”；也可以加载内置示例。
2. **提问与核对**：输入完整问题，展开参考片段查看来源、PDF 页码与原文。
3. **接入模型**：在左侧设置中选择 API 问答，分别填好问答和 Embedding 的配置并应用。
4. **重新建库**：切换 Embedding 模型或分段参数后，选齐全部资料，点击“重建知识库”。

离线演示使用确定性的字符/词语哈希向量，只能展示词面相近的内容，不能代表语义检索效果。
API 模式使用 Embedding 模型生成向量，再由大模型结合检索结果生成带编号引用的回答。
相似度是检索信号，不等于答案正确率；资料不足时系统会提示补充资料。

支持 UTF-8、带 BOM 的 UTF-16 和 GB18030 TXT。PDF 必须具有可提取文本，扫描件需先做 OCR。
本项目是单用户本机原型，不包含用户鉴权、OCR 和多人知识库隔离。
    """)
