# -*- coding: utf-8 -*-
"""
检索测试：输入问题，看能否从知识库中检索出相关片段。
"""
from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer

# ============ 路径配置 ============
BASE_DIR = Path(r"D:\HuaweiMoveData\Users\重庆森林\Desktop\法商项目")
CHROMA_DIR = BASE_DIR / "kb" / "chroma_db"

EMBED_MODEL_NAME = "BAAI/bge-small-zh-v1.5"
TOP_K = 5  # 每次检索返回最相关的几块


def main():
    print("=" * 60)
    print("检索测试")
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

        # 问题向量化
        q_vec = model.encode([question]).tolist()

        # 检索
        results = collection.query(
            query_embeddings=q_vec,
            n_results=TOP_K,
        )

        docs = results["documents"][0]
        metas = results["metadatas"][0]
        distances = results["distances"][0]

        print(f"\n{'=' * 60}")
        print(f"问题：{question}")
        print(f"{'=' * 60}")

        for i, (doc, meta, dist) in enumerate(zip(docs, metas, distances), 1):
            similarity = 1 - dist  # cosine 距离转相似度
            print(f"\n【片段 {i}】来源：{meta['source']}  类别：{meta['category']}  "
                  f"相似度：{similarity:.3f}")
            print("-" * 60)
            print(doc[:400])
            if len(doc) > 400:
                print("...（省略）")


if __name__ == "__main__":
    main()