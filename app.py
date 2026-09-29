from pathlib import Path
import sys
import json
import pickle
import re

import numpy as np


# =========================
# 0. 兼容当前 Python 环境
# =========================

SITE_PACKAGES = r"D:\Lib\site-packages"

if SITE_PACKAGES not in sys.path:
    sys.path.append(SITE_PACKAGES)


import jieba
import joblib
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

from qa_engine import (
    answer_question as engine_answer_question
)


# =========================
# 1. TF-IDF 分词函数
# 必须放在 joblib.load 之前
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
    "北汽蓝谷": ["北汽蓝谷"],
    "比亚迪": ["比亚迪"],
    "广汽": ["广汽", "广汽集团"],
    "江淮": ["江淮", "江淮汽车"],
    "赛力斯": ["赛力斯"],
    "上汽": ["上汽", "上汽集团"],
    "宇通": ["宇通", "宇通客车", "宇通集团"],
    "长安": ["长安", "长安汽车"],
    "长城": ["长城", "长城汽车"],
    "中通": ["中通", "中通客车"]
}


def detect_companies(query):

    companies = []

    for company, aliases in COMPANY_ALIASES.items():

        if any(alias in query for alias in aliases):
            companies.append(company)

    return companies


# =========================
# 3. 项目路径
# =========================

BASE_DIR = Path(__file__).resolve().parent
INDEX_DIR = BASE_DIR / "index"


# =========================
# 4. 加载知识库
# =========================

@st.cache_resource
def load_knowledge_base():

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

    return (
        bm25,
        vectorizer,
        tfidf_matrix,
        chunks
    )


bm25, vectorizer, tfidf_matrix, chunks = (
    load_knowledge_base()
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

def search(query, top_k=5):

    target_companies = detect_companies(query)

    # BM25
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

    # TF-IDF
    query_vector = vectorizer.transform(
        [query]
    )

    vector_scores = cosine_similarity(
        query_vector,
        tfidf_matrix
    ).flatten()

    # 混合得分
    final_scores = (
        0.55 * bm25_scores
        +
        0.45 * vector_scores
    )

    # 指定公司时，仅搜索对应公司
    if target_companies:

        for i, chunk in enumerate(chunks):

            if chunk.get("company") not in target_companies:
                final_scores[i] = -1

    sorted_indices = np.argsort(
        final_scores
    )[::-1]

    results = []

    for idx in sorted_indices:

        if final_scores[idx] < 0:
            continue

        results.append(
            {
                "score": float(final_scores[idx]),
                "chunk": chunks[idx]
            }
        )

        if len(results) >= top_k:
            break

    return results


# =========================
# 7. 数字处理
# =========================

def clean_number(number_text):

    return number_text.replace(",", "").strip()


def yuan_to_yi(number_text):

    try:

        value = float(
            clean_number(number_text)
        )

        return value / 100000000

    except ValueError:

        return None


# =========================
# 8. 识别研发投入问题
# =========================

def answer_rnd_investment(query, results):

    if "研发投入" not in query:
        return None

    for result in results:

        chunk = result["chunk"]
        content = chunk.get("content", "")

        for line in content.splitlines():

            if "研发投入金额" not in line:
                continue

            numbers = re.findall(
                r"\d[\d,]*\.?\d*%?",
                line
            )

            money_numbers = []
            percent_numbers = []

            for number in numbers:

                if number.endswith("%"):
                    percent_numbers.append(number)

                else:
                    money_numbers.append(number)

            if len(money_numbers) >= 2:

                current_yi = yuan_to_yi(
                    money_numbers[0]
                )

                previous_yi = yuan_to_yi(
                    money_numbers[1]
                )

                if (
                    current_yi is not None
                    and previous_yi is not None
                ):

                    company = chunk.get(
                        "company",
                        ""
                    )

                    answer = (
                        f"{company}2025年研发投入为"
                        f"{current_yi:.2f}亿元，"
                        f"2024年为"
                        f"{previous_yi:.2f}亿元"
                    )

                    if percent_numbers:

                        answer += (
                            f"，同比增长"
                            f"{percent_numbers[0]}"
                        )

                    answer += "。"

                    return {
                        "answer": answer,
                        "source": chunk
                    }

    return None


# =========================
# 9. 通用回答逻辑
# =========================

STOPWORDS = {
    "多少",
    "是什么",
    "哪些",
    "什么",
    "公司",
    "2025",
    "2024",
    "年",
    "的",
    "是多少",
    "分别",
    "情况"
}


def get_keywords(query):

    return {
        word
        for word in tokenize(query)
        if len(word) >= 2
        and word not in STOPWORDS
    }


def split_sentences(text):

    parts = re.split(
        r"[。！？；\n]+",
        text
    )

    return [
        part.strip()
        for part in parts
        if len(part.strip()) >= 8
    ]


def generic_answer(query, results):

    keywords = get_keywords(query)

    candidates = []

    for rank, result in enumerate(
        results,
        start=1
    ):

        chunk = result["chunk"]
        content = chunk.get("content", "")

        if chunk.get("type") == "table":

            pieces = [
                line.strip()
                for line in content.splitlines()
                if line.strip()
            ]

        else:

            pieces = split_sentences(content)

        for piece in pieces:

            piece_words = set(
                tokenize(piece)
            )

            overlap = len(
                keywords & piece_words
            )

            number_bonus = (
                1 if re.search(r"\d", piece)
                else 0
            )

            rank_bonus = max(
                0,
                6 - rank
            ) * 0.15

            score = (
                overlap * 2
                +
                number_bonus
                +
                rank_bonus
            )

            if score > 0:

                candidates.append(
                    {
                        "score": score,
                        "text": piece,
                        "chunk": chunk
                    }
                )

    if not candidates:
        return None

    candidates.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    best = candidates[0]

    return {
        "answer": best["text"],
        "source": best["chunk"]
    }


# =========================
# 10. 最终问答函数
# =========================

def answer_question(query):

    result = engine_answer_question(query)

    return result, result.get("results", [])


# =========================
# 11. Streamlit 页面
# =========================

st.set_page_config(
    page_title="上市公司财报问答知识库",
    page_icon="📊",
    layout="wide"
)


st.title("📊 上市公司财报问答知识库")

st.caption(
    "基于10家A股汽车上市公司2025年年度报告构建 · "
    "BM25 + TF-IDF 混合检索"
)


st.info(
    f"当前知识库共包含 {len(chunks)} 个财报文本块，"
    "回答均附原始财报出处。"
)


query = st.text_input(
    "请输入财报问题",
    placeholder="例如：比亚迪2025年研发投入是多少？"
)


ask_button = st.button(
    "提问",
    type="primary"
)


if ask_button:

    if not query.strip():

        st.warning(
            "请先输入问题。"
        )

    else:

        with st.spinner(
            "正在检索财报并生成答案..."
        ):

            answer_result, results = (
                answer_question(
                    query.strip()
                )
            )


        st.subheader("回答")


        if answer_result:

            st.success(
                answer_result["answer"]
            )

            sources = answer_result.get(
                "sources",
                []
            )

            if sources:

                st.subheader("出处")

                seen_sources = set()

                for source in sources:

                    source_key = (
                        source.get("source_file", ""),
                        source.get("page", ""),
                        source.get("section", "")
                    )

                    if source_key in seen_sources:
                        continue

                    seen_sources.add(source_key)

                    st.markdown(
                        f"""
**公司：** {source.get("company", "")}  
**来源文件：** {source.get("source_file", "")}  
**页码：** 第 {source.get("page", "")} 页  
**章节：** {source.get("section", "")}
"""
                    )

            else:

                st.caption(
                    "本次回答未定位到唯一出处。"
                )

        else:

            st.warning(
                "暂未从财报中找到足够明确的答案。"
            )


        # 后台召回记录
        with st.expander(
            "查看后台召回记录"
        ):

            for i, result in enumerate(
                results,
                start=1
            ):

                chunk = result["chunk"]

                st.markdown(
                    f"""
### 召回 {i}

- **公司：** {chunk.get("company", "")}
- **页码：** 第 {chunk.get("page", "")} 页
- **章节：** {chunk.get("section", "")}
- **类型：** {chunk.get("type", "")}
- **综合得分：** {result["score"]:.4f}
"""
                )

                st.text(
                    chunk.get(
                        "content",
                        ""
                    )[:1200]
                )

                st.divider()
