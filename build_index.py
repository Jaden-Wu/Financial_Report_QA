from pathlib import Path
import sys
import json
import pickle

SITE_PACKAGES = r"D:\Lib\site-packages"

if SITE_PACKAGES not in sys.path:
    sys.path.append(SITE_PACKAGES)

import jieba

from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib


# =========================
# 1. 项目路径
# =========================

BASE_DIR = Path(__file__).resolve().parent

CHUNKS_FILE = BASE_DIR / "processed" / "chunks.jsonl"
INDEX_DIR = BASE_DIR / "index"

INDEX_DIR.mkdir(exist_ok=True)


# =========================
# 2. 读取 chunks
# =========================

chunks = []

with CHUNKS_FILE.open("r", encoding="utf-8") as f:
    for line in f:
        chunks.append(json.loads(line))

print(f"共读取 {len(chunks)} 个 chunk。")


# =========================
# 3. 整理检索文本
# =========================

search_texts = []

for chunk in chunks:

    company = chunk.get("company", "")
    section = chunk.get("section", "")
    content = chunk.get("content", "")

    text = (
        f"公司：{company}\n"
        f"章节：{section}\n"
        f"{content}"
    )

    search_texts.append(text)


# =========================
# 4. BM25
# =========================

print("正在建立 BM25 索引...")


def tokenize(text):
    return [
        word.strip()
        for word in jieba.lcut(text)
        if word.strip()
    ]


tokenized_corpus = [
    tokenize(text)
    for text in search_texts
]

bm25 = BM25Okapi(tokenized_corpus)

with (INDEX_DIR / "bm25.pkl").open("wb") as f:
    pickle.dump(bm25, f)

print("BM25 索引完成。")


# =========================
# 5. TF-IDF 向量索引
# =========================

print("正在建立 TF-IDF 向量索引...")


def jieba_tokenizer(text):
    return [
        word.strip()
        for word in jieba.lcut(text)
        if word.strip()
    ]


vectorizer = TfidfVectorizer(
    tokenizer=jieba_tokenizer,
    token_pattern=None,
    max_features=50000,
    sublinear_tf=True
)

tfidf_matrix = vectorizer.fit_transform(search_texts)


joblib.dump(
    vectorizer,
    INDEX_DIR / "tfidf_vectorizer.joblib"
)

joblib.dump(
    tfidf_matrix,
    INDEX_DIR / "tfidf_matrix.joblib"
)


print("TF-IDF 向量索引完成。")


# =========================
# 6. 保存 chunk 元数据
# =========================

with (INDEX_DIR / "chunks.json").open(
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        chunks,
        f,
        ensure_ascii=False
    )


print()
print("=" * 50)
print("索引全部建立完成。")
print(f"Chunk 数量：{len(chunks)}")
print(f"向量维度：{tfidf_matrix.shape[1]}")
print(f"索引目录：{INDEX_DIR}")
print("=" * 50)