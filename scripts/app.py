# -*- coding: utf-8 -*-
"""
法商结合出海顾问 · Streamlit 网页版 v2（非流式，稳定版）
"""
import json
import requests
from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer
import streamlit as st

# ============ 路径 ============
BASE_DIR = Path(r"D:\HuaweiMoveData\Users\重庆森林\Desktop\法商项目")
CHROMA_DIR = BASE_DIR / "kb" / "chroma_db"
EMBED_MODEL_NAME = "BAAI/bge-small-zh-v1.5"

API_URL = "https://api.deepseek.com/chat/completions"

SYSTEM_PROMPT = """你是法商结合出海合规顾问，正在协助用户研究长江存储的出海合规问题。

你有两类参考资料：
1. 企业资料：长江存储 IPO 申请书
2. 法律资料：出口管制、实体清单、两用物项管理等法规

回答要求：
- 企业事实必须来自企业资料，法律依据必须来自法律文件。
- 引用具体来源（文件名 + 章节或条款号）。
- 做交叉分析时，明确区分"企业事实"和"法律要求"。
- 资料中没有的信息，如实说明"资料不足"，不要编造。
- 不做正式法律意见，只做资料梳理和分析辅助。
- 回答尽量控制在 800 字以内。
"""


# ============ 缓存资源 ============
@st.cache_resource(show_spinner=False)
def load_model():
    return SentenceTransformer(EMBED_MODEL_NAME)


@st.cache_resource(show_spinner=False)
def load_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_collection("knowledge_base")


# ============ 检索 ============
def _text_overlap(a, b):
    a_set = set(a[:200])
    b_set = set(b[:200])
    if not a_set or not b_set:
        return 0
    return len(a_set & b_set) / len(a_set | b_set)


def retrieve_balanced(collection, model, question, k_legal=3, k_enterprise=2):
    q_vec = model.encode([question]).tolist()

    legal_results = collection.query(
        query_embeddings=q_vec, n_results=k_legal, where={"category": "法律"},
    )
    enterprise_results = collection.query(
        query_embeddings=q_vec, n_results=k_enterprise, where={"category": "企业"},
    )

    all_items = []
    for results in [legal_results, enterprise_results]:
        for d, m, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
            all_items.append((d, m, 1 - dist))

    deduped = []
    for d, m, sim in all_items:
        is_dup = False
        for kept_doc, _, _ in deduped:
            if _text_overlap(d, kept_doc) > 0.9:
                is_dup = True
                break
        if not is_dup:
            deduped.append((d, m, sim))

    return deduped


def build_context(deduped):
    parts = []
    for i, (doc, meta, sim) in enumerate(deduped, 1):
        parts.append(
            f"【片段 {i}】来源：{meta['source']}（{meta['category']}）  相似度：{sim:.3f}\n"
            f"{doc}"
        )
    return "\n\n".join(parts)


# ============ 调用 DeepSeek（非流式） ============
def call_deepseek(question, context, api_key, model_name):
    user_prompt = f"""请根据下面的参考资料回答问题。

【参考资料】
{context}

【问题】
{question}

请给出结构化的回答：
1. 先给出结论
2. 再列出依据（引用具体来源）
3. 如果是交叉问题，分别说明企业事实和法律依据
4. 资料不足时如实说明
"""
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.3,
    }
    resp = requests.post(API_URL, headers=headers, json=payload, timeout=120)
    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:300]}")
    return resp.json()["choices"][0]["message"]["content"]


# ============ 页面 ============
st.set_page_config(page_title="法商结合出海顾问", page_icon="⚖️", layout="wide")

with st.sidebar:
    st.header("⚙️ 配置")
    api_key = st.text_input("DeepSeek API Key", type="password",
                            help="不会被保存，仅本次会话使用")
    model_name = st.selectbox("模型", ["deepseek-chat", "deepseek-reasoner"], index=0,
                              help="chat 更快，reasoner 推理更强但更慢、更贵")
    st.divider()
    k_legal = st.slider("法律片段数", 1, 5, 3)
    k_enterprise = st.slider("企业片段数", 1, 5, 2)
    st.divider()
    st.caption("💡 本工具仅做资料梳理与分析辅助，不构成正式法律意见。")
    if st.button("🗑️ 清空对话"):
        st.session_state.messages = []
        st.rerun()

st.title("⚖️ 法商结合出海顾问")
st.caption("基于长江存储 IPO 申请书 + 出口管制法律文件的知识库问答")

if "messages" not in st.session_state:
    st.session_state.messages = []

# 展示历史
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander(f"📚 引用来源（{len(msg['sources'])} 个片段）"):
                for i, (doc, meta, sim) in enumerate(msg["sources"], 1):
                    st.markdown(f"**{i}. {meta['source']}**（{meta['category']}）  相似度 {sim:.3f}")
                    st.text(doc[:500] + ("..." if len(doc) > 500 else ""))

# 输入
question = st.chat_input("请输入你的问题...")

if question:
    if not api_key:
        st.error("请先在左侧填写 DeepSeek API Key。")
        st.stop()

    # 先把用户消息存入历史
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("正在检索知识库..."):
            model = load_model()
            collection = load_collection()
            deduped = retrieve_balanced(collection, model, question, k_legal, k_enterprise)
            context = build_context(deduped)

        with st.spinner("AI 正在生成回答..."):
            try:
                answer = call_deepseek(question, context, api_key, model_name)
            except Exception as e:
                answer = f"⚠️ 调用失败：{e}"

        st.markdown(answer)

        with st.expander(f"📚 引用来源（{len(deduped)} 个片段）"):
            for i, (doc, meta, sim) in enumerate(deduped, 1):
                st.markdown(f"**{i}. {meta['source']}**（{meta['category']}）  相似度 {sim:.3f}")
                st.text(doc[:500] + ("..." if len(doc) > 500 else ""))

    # 再存 AI 消息
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": deduped,
    })