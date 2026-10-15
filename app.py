"""Traditional Chinese Streamlit interface for the OpenAI RAG demo."""
import os
from dotenv import load_dotenv
import streamlit as st
from openai import OpenAI
from rag.core import Retriever, answer_question, chunk_text, read_document, NO_ANSWER

load_dotenv()
st.set_page_config(page_title="企業知識庫 RAG 問答系統", page_icon="📚", layout="wide")
st.title("📚 企業知識庫 RAG 問答系統")
st.caption("OpenAI Embeddings 語意檢索 × 文件引用 × AI 問答")
st.info("示範專案：請只使用虛構或已獲授權的文件；文件內容會傳送至 OpenAI API。")

with st.sidebar:
    st.header("系統設定")
    api_key = st.text_input("OpenAI API Key", value=os.getenv("OPENAI_API_KEY", ""), type="password")
    top_k = st.slider("檢索片段數（Top-K）", 1, 8, 3)
    threshold = st.slider("最低相似度", -1.0, 1.0, 0.30, 0.05, help="Cosine similarity 門檻；需依實際資料測試調整，不保證完全過濾無關內容。")
    st.caption("向量模型：gemini-embedding-001")
    st.caption("問答模型：gemini-3.5-flash-lite")

st.subheader("1. 建立知識庫")
uploads = st.file_uploader("上傳公司知識文件（可多選）", type=["txt", "md", "pdf"], accept_multiple_files=True)
st.caption("支援 TXT、Markdown、具文字層的 PDF；掃描型 PDF 請先進行 OCR。")
if st.button("建立向量知識庫", type="primary", disabled=not uploads):
    if not api_key.strip():
        st.error("請先設定 OpenAI API Key。")
    else:
        chunks, failures = [], []
        for file in uploads:
            try:
                content = read_document(file.name, file.getvalue())
                parts = chunk_text(content, file.name)
                if not parts:
                    raise ValueError("沒有可用文字，掃描 PDF 可能需要 OCR")
                chunks.extend(parts)
            except Exception as exc:
                failures.append(f"{file.name}：{exc}")
        for failure in failures:
            st.warning(f"無法處理 {failure}")
        if chunks:
            try:
                with st.spinner("正在透過 OpenAI API 建立文件向量…"):
                    client = OpenAI(api_key=api_key, timeout=30.0, max_retries=2,base_url="https://generativelanguage.googleapis.com/v1beta/openai/")
                    retriever = Retriever(chunks, client)
                st.session_state["retriever"] = retriever
                st.session_state["indexed_files"] = sorted({c.source for c in chunks})
                st.session_state.pop("latest_answer", None)
                st.success(f"知識庫建立完成：{len(chunks)} 個片段、{len(st.session_state['indexed_files'])} 份文件。")
            except Exception as exc:
                st.session_state.pop("retriever", None)
                st.error(f"建立向量知識庫失敗：{type(exc).__name__}：{exc}")
        else:
            st.error("沒有可建立索引的文字內容。")

st.divider()
st.subheader("2. 文件問答")
if "retriever" not in st.session_state:
    st.info("請先上傳文件並建立向量知識庫。")
else:
    st.caption("已建立索引：" + "、".join(st.session_state["indexed_files"]))
    question = st.text_input("輸入你想詢問的問題", placeholder="例如：員工每年有幾天特休假？")
    if st.button("搜尋並產生回答", disabled=not question.strip()):
        if not api_key.strip():
            st.error("請先設定 OpenAI API Key。")
        else:
            try:
                with st.spinner("正在檢索相關文件並產生回答…"):
                    # Use the same API credential provided when building the index.
                    retriever = st.session_state["retriever"]
                    retriever.client = OpenAI(api_key=api_key, timeout=30.0, max_retries=2,base_url="https://generativelanguage.googleapis.com/v1beta/openai/")
                    hits = retriever.search(question, top_k=top_k, threshold=threshold)
                    answer = answer_question(question, hits, retriever.client) if hits else NO_ANSWER
                st.session_state["latest_answer"] = (answer, hits)
            except Exception as exc:
                st.session_state.pop("latest_answer", None)
                st.error(f"問答失敗：{type(exc).__name__}：{exc}")
    if "latest_answer" in st.session_state:
        answer, hits = st.session_state["latest_answer"]
        st.markdown("#### AI 回答")
        st.write(answer)
        st.markdown("#### 引用文件片段")
        if not hits:
            st.warning("沒有通過相似度門檻的文件片段。可重新提問或適度調整門檻。")
        for i, (chunk, score) in enumerate(hits, 1):
            with st.expander(f"[S{i}] {chunk.source}｜片段 {chunk.chunk_id}｜相似度 {score:.3f}"):
                st.write(chunk.text)

st.divider()
st.caption("檢索結果及 AI 回答可能出錯；重要資訊請回到原始文件核對。上傳新文件後請重新建立索引。")
