from pathlib import Path
import sys
import json
import pickle

import numpy as np


# =========================
# 0. 兼容你当前的 Python 环境
# =========================

SITE_PACKAGES = r"D:\Lib\site-packages"

if SITE_PACKAGES not in sys.path:
    sys.path.append(SITE_PACKAGES)


import jieba
import joblib
from sklearn.metrics.pairwise import cosine_similarity


# =========================
# 1. TF-IDF 分词函数
# 必须写在 joblib.load 之前
# =========================

def jieba_tokenizer(text):
    return [
        word.strip()
        for word in jieba.lcut(text)
        if word.strip()
    ]


# =========================
# 2. 公司名称与别名
# =========================

COMPANY_ALIASES = {
    "北汽蓝谷": [
        "北汽蓝谷"
    ],

    "比亚迪": [
        "比亚迪"
    ],

    "广汽": [
        "广汽",
        "广汽集团"
    ],

    "江淮": [
        "江淮",
        "江淮汽车"
    ],

    "赛力斯": [
        "赛力斯"
    ],

    "上汽": [
        "上汽",
        "上汽集团"
    ],

    "宇通": [
        "宇通",
        "宇通客车",
        "宇通集团"
    ],

    "长安": [
        "长安",
        "长安汽车"
    ],

    "长城": [
        "长城",
        "长城汽车"
    ],

    "中通": [
        "中通",
        "中通客车"
    ]
}


def detect_companies(query):
    """
    检测用户问题里是否明确提到了某一家或多家公司。
    """

    companies = []

    for company, aliases in COMPANY_ALIASES.items():

        for alias in aliases:

            if alias in query:
                companies.append(company)
                break

    return companies


# =========================
# 3. 项目路径
# =========================

BASE_DIR = Path(__file__).resolve().parent
INDEX_DIR = BASE_DIR / "index"


# =========================
# 4. 加载索引
# =========================

print("正在加载索引...")


with (INDEX_DIR / "bm25.pkl").open("rb") as f:
    bm25 = pickle.load(f)


vectorizer = joblib.load(
    INDEX_DIR / "tfidf_vectorizer.joblib"
)


tfidf_matrix = joblib.load(
    INDEX_DIR / "tfidf_matrix.joblib"
)


with (INDEX_DIR / "chunks.json").open(
    "r",
    encoding="utf-8"
) as f:

    chunks = json.load(f)


print(
    f"索引加载完成，共 {len(chunks)} 个 chunk。"
)


# =========================
# 5. 中文分词
# =========================

def tokenize(text):
    return [
        word.strip()
        for word in jieba.lcut(text)
        if word.strip()
    ]


# =========================
# 6. BM25 + TF-IDF 混合检索
# =========================

def search(query, top_k=8):

    # -------------------------
    # 检测问题中指定了哪些公司
    # -------------------------

    target_companies = detect_companies(query)

    if target_companies:
        print(
            "已识别目标公司："
            + "、".join(target_companies)
        )

    else:
        print(
            "未指定具体公司，将在全部公司中检索。"
        )


    # -------------------------
    # BM25
    # -------------------------

    query_tokens = tokenize(query)

    bm25_scores = np.array(
        bm25.get_scores(query_tokens),
        dtype=float
    )

    if bm25_scores.max() > 0:

        bm25_scores = (
            bm25_scores
            / bm25_scores.max()
        )


    # -------------------------
    # TF-IDF 向量检索
    # -------------------------

    query_vector = vectorizer.transform(
        [query]
    )

    tfidf_scores = cosine_similarity(
        query_vector,
        tfidf_matrix
    ).flatten()


    # -------------------------
    # 混合得分
    # -------------------------

    final_scores = (
        0.5 * bm25_scores
        +
        0.5 * tfidf_scores
    )


    # -------------------------
    # 如果明确指定公司
    # 只保留指定公司的结果
    # -------------------------

    if target_companies:

        for i, chunk in enumerate(chunks):

            company = chunk.get(
                "company",
                ""
            )

            if company not in target_companies:

                final_scores[i] = -1


    # -------------------------
    # 排序取 Top K
    # -------------------------

    sorted_indices = np.argsort(
        final_scores
    )[::-1]


    results = []


    for idx in sorted_indices:

        # 被过滤掉的公司不显示
        if final_scores[idx] < 0:
            continue

        chunk = chunks[idx]

        results.append(
            {
                "rank": len(results) + 1,

                "score": float(
                    final_scores[idx]
                ),

                "bm25_score": float(
                    bm25_scores[idx]
                ),

                "vector_score": float(
                    tfidf_scores[idx]
                ),

                "chunk": chunk
            }
        )

        if len(results) >= top_k:
            break


    return results


# =========================
# 7. 输入问题
# =========================

query = input(
    "\n请输入财报问题："
).strip()


if not query:
    raise ValueError(
        "问题不能为空。"
    )


# =========================
# 8. 执行检索
# =========================

results = search(
    query,
    top_k=8
)


# =========================
# 9. 显示召回结果
# =========================

print("\n")

print("=" * 70)

print(
    f"问题：{query}"
)

print("=" * 70)


for result in results:

    chunk = result["chunk"]

    print()

    print(
        f"【结果 {result['rank']}】"
    )

    print(
        f"综合得分："
        f"{result['score']:.4f}"
        f" | BM25："
        f"{result['bm25_score']:.4f}"
        f" | 向量："
        f"{result['vector_score']:.4f}"
    )

    print(
        f"公司："
        f"{chunk.get('company', '')}"
    )

    print(
        f"来源："
        f"{chunk.get('source_file', '')}"
    )

    print(
        f"页码："
        f"{chunk.get('page', '')}"
    )

    print(
        f"章节："
        f"{chunk.get('section', '')}"
    )

    print(
        f"类型："
        f"{chunk.get('type', '')}"
    )

    print("原文：")

    print(
        chunk.get(
            "content",
            ""
        )[:1200]
    )

    print(
        "-" * 70
    )