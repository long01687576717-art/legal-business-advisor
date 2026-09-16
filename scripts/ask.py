# -*- coding: utf-8 -*-
"""
完整问答 v2：分类检索 + 去重 + 调 DeepSeek API。
"""
import requests
from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer

# ============ 配置 ============
BASE_DIR = Path(r"D:\HuaweiMoveData\Users\重庆森林\Desktop\法商项目")
CHROMA_DIR = BASE_DIR / "kb" / "chroma_db"
EMBED_MODEL_NAME = "BAAI/bge-small-zh-v1.5"

# 分类检索数量
TOP_K_LEGAL = 3      # 法律取 3 块
TOP_K_ENTERPRISE = 2 # 企业取 2 块
DEDUP_THRESHOLD = 0.9  # 相似度超过这个值视为重复

API_URL = "https://api.deepseek.com/chat/completions"
MODEL_NAME = "deepseek-chat"
API_KEY = "sk-c936d5cd2acc4fc99215b2127b2a6a86"


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


def retrieve_balanced(collection, model, question):
    """分类检索：法律和企业各取固定数量，然后去重。"""
    q_vec = model.encode([question]).tolist()

    # 法律检索
    legal_results = collection.query(
        query_embeddings=q_vec,
        n_results=TOP_K_LEGAL,
        where={"category": "法律"},
    )

    # 企业检索
    enterprise_results = collection.query(
        query_embeddings=q_vec,
        n_results=TOP_K_ENTERPRISE,
        where={"category": "企业"},
    )

    # 合并
    all_docs = []
    all_metas = []
    all_dists = []

    for docs, metas, dists in [
        (legal_results["documents"][0], legal_results["metadatas"][0], legal_results["distances"][0]),
        (enterprise_results["documents"][0], enterprise_results["metadatas"][0], enterprise_results["distances"][0]),
    ]:
        for d, m, dist in zip(docs, metas, dists):
            all_docs.append(d)
            all_metas.append(m)
            all_dists.append(dist)

    # 去重：相似度（1 - distance）超过阈值视为重复
    deduped = []
    seen_similarities = []
    for d, m, dist in zip(all_docs, all_metas, all_dists):
        sim = 1 - dist
        # 如果和已保留的某个片段相似度超过阈值，跳过
        is_dup = False
        for kept_doc, kept_sim in seen_similarities:
            # 简单判断：如果两段文本前 100 字高度重叠，或相似度都极高，视为重复
            if _text_overlap(d, kept_doc) > DEDUP_THRESHOLD:
                is_dup = True
                break
        if not is_dup:
            deduped.append((d, m, sim))
            seen_similarities.append((d, sim))

    return deduped


def _text_overlap(a, b):
    """粗略判断两段文本的重叠程度：取各自前 200 字，算字符级 Jaccard 相似度。"""
    a_set = set(a[:200])
    b_set = set(b[:200])
    if not a_set or not b_set:
        return 0
    return len(a_set & b_set) / len(a_set | b_set)


def build_context(deduped):
    """把去重后的片段拼成上下文。"""
    parts = []
    for i, (doc, meta, sim) in enumerate(deduped, 1):
        parts.append(
            f"【片段 {i}】来源：{meta['source']}（{meta['category']}）  相似度：{sim:.3f}\n"
            f"{doc}"
        )
    return "\n\n".join(parts)


def call_deepseek(question, context, api_key):
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
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.3,
    }
    resp = requests.post(API_URL, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def main():
    if not API_KEY or API_KEY == "sk-你的Key":
        print("请先在代码里填写真实的 API Key。")
        return

    print("=" * 60)
    print("法商结合出海顾问 v2（分类检索 + 去重）")
    print("=" * 60)

    print("\n[1/2] 加载嵌入模型...")
    model = SentenceTransformer(EMBED_MODEL_NAME)

    print("[2/2] 连接知识库...")
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_collection("knowledge_base")
    print(f"知识库共 {collection.count()} 块。\n")

    while True:
        question = input("\n请输入问题（输入 q 退出）：").strip()
        if question.lower() == "q":
            break
        if not question:
            continue

        deduped = retrieve_balanced(collection, model, question)
        context = build_context(deduped)

        print(f"\n检索到 {len(deduped)} 个去重后片段。")
        print("正在生成回答...")
        try:
            answer = call_deepseek(question, context, API_KEY)
            print("\n" + "=" * 60)
            print("回答：")
            print("=" * 60)
            print(answer)
        except Exception as e:
            print(f"\n调用失败：{e}")

        print("\n" + "-" * 60)
        print("引用来源：")
        for i, (doc, meta, sim) in enumerate(deduped, 1):
            print(f"  {i}. {meta['source']}（{meta['category']}）  相似度 {sim:.3f}")


if __name__ == "__main__":
    main()