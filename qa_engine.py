from pathlib import Path
import sys
import json
import pickle
import re

SITE_PACKAGES = r"D:\Lib\site-packages"

if SITE_PACKAGES not in sys.path:
    sys.path.append(SITE_PACKAGES)

import numpy as np
import jieba
import joblib
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# 1. 分词函数
# =========================================================

def jieba_tokenizer(text):
    return [
        word.strip()
        for word in jieba.lcut(text)
        if word.strip()
    ]


def tokenize(text):
    return [
        word.strip()
        for word in jieba.lcut(text)
        if word.strip()
    ]


# 旧版索引在 build_index.py 作为主程序运行时生成，joblib 因而把
# jieba_tokenizer 记录在 __main__ 下。注册这个兼容别名后，qa_engine
# 既可以直接运行，也可以被 Streamlit 和测试代码安全导入。
main_module = sys.modules.get("__main__")

if (
    main_module is not None
    and not hasattr(main_module, "jieba_tokenizer")
):
    setattr(
        main_module,
        "jieba_tokenizer",
        jieba_tokenizer
    )


# =========================================================
# 2. 公司名称
# =========================================================

COMPANY_ALIASES = {
    "北汽蓝谷": ["北汽蓝谷"],
    "比亚迪": ["比亚迪"],
    "广汽": ["广汽集团", "广汽"],
    "江淮": ["江淮汽车", "江淮"],
    "赛力斯": ["赛力斯"],
    "上汽": ["上汽集团", "上汽"],
    "宇通": ["宇通集团", "宇通客车", "宇通"],
    "长安": ["长安汽车", "长安"],
    "长城": ["长城汽车", "长城"],
    "中通": ["中通客车", "中通"]
}


def detect_companies(query):

    matches = []

    for company, aliases in COMPANY_ALIASES.items():

        positions = []

        for alias in aliases:

            pos = query.find(alias)

            if pos >= 0:
                positions.append(pos)

        if positions:

            matches.append(
                (min(positions), company)
            )

    matches.sort(
        key=lambda x: x[0]
    )

    return [
        company
        for _, company in matches
    ]


# =========================================================
# 3. 指标
# =========================================================

METRICS = {

    "rnd": {
        "name": "研发投入",
        "labels": [
            "研发投入金额（元）",
            "研发投入金额(元)",
            "研发投入金额",
            "研发投入合计",
            "研发投入总额"
        ]
    },

    "revenue": {
        "name": "营业收入",
        "labels": [
            "营业收入"
        ]
    },

    "net_profit": {
        "name": "归属于上市公司股东的净利润",
        "labels": [
            "归属于上市公司股东的净利润",
            "归属于母公司所有者的净利润",
            "归属于母公司股东的净利润"
        ]
    },

    "operating_cashflow": {
        "name": "经营活动产生的现金流量净额",
        "labels": [
            "经营活动产生的现金流量净额"
        ]
    }
}


def detect_metric(query):

    if "研发投入" in query:
        return "rnd"

    if (
        "归属于上市公司股东的净利润" in query
        or "归母净利润" in query
        or "归属于母公司所有者的净利润" in query
    ):
        return "net_profit"

    if (
        "经营活动产生的现金流量净额" in query
        or "经营活动现金流量净额" in query
        or "经营现金流" in query
    ):
        return "operating_cashflow"

    if "营业收入" in query:
        return "revenue"

    return None


# =========================================================
# 4. 加载知识库
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
INDEX_DIR = BASE_DIR / "index"

print("正在加载财报知识库...")


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


# 表格 chunk 本身通常不保留“单位：万元/元”。从 processed 中建立
# 页级上下文，精确命中后再回到原页判断单位，避免把万元当成元。
PAGE_CONTEXTS = {}
PROCESSED_DIR = BASE_DIR / "processed"

for processed_file in PROCESSED_DIR.glob("*.jsonl"):

    if processed_file.name == "chunks.jsonl":
        continue

    with processed_file.open(
        "r",
        encoding="utf-8"
    ) as f:

        for record_line in f:

            record = json.loads(record_line)
            key = (
                record.get("company"),
                record.get("page")
            )

            PAGE_CONTEXTS[key] = record.get(
                "text",
                ""
            )


print(
    f"知识库加载完成，共 {len(chunks)} 个文本块。"
)


# =========================================================
# 5. 普通混合检索
# =========================================================

def search(query, company=None, top_k=10):

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


    query_vector = vectorizer.transform(
        [query]
    )

    vector_scores = cosine_similarity(
        query_vector,
        tfidf_matrix
    ).flatten()


    final_scores = (
        0.55 * bm25_scores
        +
        0.45 * vector_scores
    )


    if company is not None:

        for i, chunk in enumerate(chunks):

            if chunk.get("company") != company:
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


# =========================================================
# 6. 表格处理
# =========================================================

def split_cells(line):

    return [
        cell.strip()
        for cell in line.split("|")
        if cell.strip()
    ]


def normalize_label(text):

    return (
        text
        .replace(" ", "")
        .replace("\n", "")
        .strip()
    )


def canonical_label(text):

    text = normalize_label(text)

    # 去除利润表行常见的序号或“其中：”，但不做模糊包含匹配，
    # 因而“营业外收入”不会被当成“营业收入”。
    text = re.sub(
        r"^(?:[一二三四五六七八九十]+、|\(?\d+\)?[\.、])",
        "",
        text
    )
    text = re.sub(
        r"^(?:其中|加|减)[：:]",
        "",
        text
    )
    text = re.sub(
        r"[（(](?:人民币)?(?:亿元|万元|元)[）)]$",
        "",
        text
    )

    return text


def row_matches_metric(line, metric):

    cells = split_cells(line)

    if not cells:
        return False

    first_cell = canonical_label(
        cells[0]
    )

    for label in METRICS[metric]["labels"]:

        expected = canonical_label(label)

        if first_cell == expected:
            return True

        # PDF 表格偶尔把长指标名截断到下一行。仅对足够长的前缀
        # 放宽匹配，短标签（尤其“营业收入”）仍坚持完全匹配。
        if (
            len(first_cell) >= 8
            and (
                expected.startswith(first_cell)
                or first_cell.startswith(expected)
            )
        ):
            return True

    return False


# =========================================================
# 7. 数字提取
# =========================================================

NUMBER_PATTERN = re.compile(
    r"^\(?-?[\d,]+(?:\.\d+)?\)?%?$"
)


def parse_number(text):

    text = text.strip()

    negative = False

    if (
        text.startswith("(")
        and text.endswith(")")
    ):

        negative = True
        text = text[1:-1]


    text = (
        text
        .replace(",", "")
        .replace("%", "")
    )


    try:

        value = float(text)

        if negative:
            value = -value

        return value

    except ValueError:

        return None


def extract_row_values(line):

    cells = split_cells(line)

    values = []
    percentages = []


    for cell in cells[1:]:

        compact = cell.replace(" ", "")

        if not NUMBER_PATTERN.fullmatch(
            compact
        ):
            continue


        if compact.endswith("%"):

            percentages.append(
                compact
            )

        else:

            value = parse_number(
                compact
            )

            if value is not None:
                values.append(value)


    return values, percentages


# =========================================================
# 8. 金额单位
# =========================================================

def detect_unit(
    content,
    line,
    page_text=""
):

    compact_line = normalize_label(line)

    # 行名中的单位最可靠。
    if (
        "（亿元）" in compact_line
        or "(亿元)" in compact_line
    ):
        return "亿元"

    if (
        "（万元）" in compact_line
        or "(万元)" in compact_line
    ):
        return "万元"

    if (
        "（元）" in compact_line
        or "(元)" in compact_line
        or "金额（元）" in compact_line
        or "金额(元)" in compact_line
    ):
        return "元"


    # 表格 chunk 未携带单位时，使用同页正文中、该指标之前最近一次
    # 出现的“单位：...”声明。
    compact_page = re.sub(
        r"\s+",
        "",
        page_text
    )
    row_label = normalize_label(
        split_cells(line)[0]
    )
    row_position = compact_page.find(
        row_label
    )

    if row_position >= 0:
        unit_scope = compact_page[
            max(0, row_position - 2500):row_position
        ]
    else:
        unit_scope = compact_page[:2500]

    unit_matches = list(
        re.finditer(
            r"单位[：:](?:人民币)?(亿元|万元|元)",
            unit_scope
        )
    )

    if unit_matches:
        return unit_matches[-1].group(1)

    combined = content[:2000] + "\n" + line


    if (
        "单位：亿元" in combined
        or "单位:亿元" in combined
    ):
        return "亿元"


    if (
        "单位：万元" in combined
        or "单位:万元" in combined
    ):
        return "万元"


    if (
        "单位：元" in combined
        or "单位:元" in combined
        or "金额（元）" in line
        or "金额(元)" in line
    ):
        return "元"


    return "unknown"


def metric_candidate_score(
    chunk,
    content,
    line,
    metric,
    values,
    percentages
):

    section = normalize_label(
        chunk.get("section", "")
    )
    compact_content = normalize_label(
        content
    )
    page_text = PAGE_CONTEXTS.get(
        (
            chunk.get("company"),
            chunk.get("page")
        ),
        ""
    )
    compact_page = normalize_label(
        page_text
    )
    label = canonical_label(
        split_cells(line)[0]
    )

    score = 0

    # 公司整体年度口径的最高优先来源。
    if (
        "主要会计数据" in section
        or "主要财务指标" in section
    ):
        score += 320

    if (
        "主要会计数据" in compact_page
        or "主要财务指标" in compact_page
    ):
        score += 120

    if (
        "2025年" in compact_content
        and "2024年" in compact_content
    ):
        score += 45

    if (
        "2025年度" in compact_content
        and "2024年度" in compact_content
    ):
        score += 55

    if "经审计" in compact_content:
        score += 20

    if metric == "rnd":

        if label in {
            "研发投入金额",
            "研发投入合计",
            "研发投入总额"
        }:
            score += 260

        if (
            "研发投入情况" in compact_page
            or "研发投入情况" in compact_content
        ):
            score += 80

    elif "财务报表" in section:
        score += 45

    # 明确排除季度、分部、子公司和其他主体等局部口径。
    if (
        "季度" in section
        or "第一季度" in compact_content
        or "第二季度" in compact_content
    ):
        score -= 600

    for local_scope_word in (
        "分部",
        "在其他主体中的权益",
        "联营企业",
        "合营企业",
        "子公司"
    ):
        if (
            local_scope_word in section
            or local_scope_word in compact_page[:1200]
        ):
            score -= 260

    # 数字丰富度只作为同口径候选之间的弱排序因素。
    score += min(len(values), 4) * 3
    score += min(len(percentages), 2)

    return score


def format_amount(value, unit):

    if unit == "元":

        yi = value / 100000000

        return f"{yi:.2f}亿元"


    if unit == "万元":

        yi = value / 10000

        return f"{yi:.2f}亿元"


    if unit == "亿元":

        return f"{value:.2f}亿元"


    # 财报里的大金额如果没有成功识别单位，
    # 这种数量级通常就是元
    if abs(value) >= 100000000:

        yi = value / 100000000

        return f"{yi:.2f}亿元"


    return f"{value:,.2f}"


# =========================================================
# 9. 核心修复：
#    不依赖 Top-K
#    直接扫描该公司全部 table chunk
# =========================================================

def find_metric_exact(company, metric):

    candidates = []


    for chunk in chunks:

        if chunk.get("company") != company:
            continue

        if chunk.get("type") != "table":
            continue


        content = chunk.get(
            "content",
            ""
        )


        for line in content.splitlines():

            if not row_matches_metric(
                line,
                metric
            ):
                continue


            values, percentages = (
                extract_row_values(line)
            )


            if not values:
                continue


            unit = detect_unit(
                content,
                line,
                PAGE_CONTEXTS.get(
                    (
                        chunk.get("company"),
                        chunk.get("page")
                    ),
                    ""
                )
            )

            priority = metric_candidate_score(
                chunk,
                content,
                line,
                metric,
                values,
                percentages
            )


            candidates.append(
                {
                    "chunk": chunk,
                    "line": line,
                    "values": values,
                    "percentages": percentages,
                    "unit": unit,
                    "priority": priority
                }
            )


    if not candidates:
        return None


    # 先按口径优先级选择，再用数字数量和页码稳定排序。
    candidates.sort(
        key=lambda x: (
            x["priority"],
            len(x["values"]),
            len(x["percentages"]),
            -int(x["chunk"].get("page") or 0)
        ),
        reverse=True
    )


    return candidates[0]


# =========================================================
# 10. 构造标准财务指标回答
# =========================================================

def build_metric_answer(
    company,
    metric,
    evidence
):

    metric_name = (
        METRICS[metric]["name"]
    )

    values = evidence["values"]

    percentages = evidence[
        "percentages"
    ]

    unit = evidence["unit"]


    current = format_amount(
        values[0],
        unit
    )


    answer = (
        f"{company}2025年"
        f"{metric_name}为"
        f"{current}"
    )


    if len(values) >= 2:

        previous = format_amount(
            values[1],
            unit
        )

        answer += (
            f"，2024年为"
            f"{previous}"
        )


    if percentages:

        pct = percentages[0]

        # 表格本身已经可能带负号
        answer += (
            f"，同比变动"
            f"{pct}"
        )


    answer += "。"

    return answer


# =========================================================
# 11. 通用问题备用逻辑
# =========================================================

def generic_answer(query, results):

    query_words = set(
        tokenize(query)
    )

    best = None
    best_score = -1


    for rank, result in enumerate(
        results,
        start=1
    ):

        chunk = result["chunk"]

        content = chunk.get(
            "content",
            ""
        )


        if chunk.get("type") == "table":

            pieces = [
                x.strip()
                for x in content.splitlines()
                if x.strip()
            ]

        else:

            pieces = re.split(
                r"[。！？；\n]+",
                content
            )


        for piece in pieces:

            if len(piece.strip()) < 8:
                continue


            words = set(
                tokenize(piece)
            )

            overlap = len(
                query_words & words
            )


            score = (
                overlap * 2
                + result["score"]
                - rank * 0.01
            )


            if score > best_score:

                best_score = score

                best = {
                    "answer": piece.strip(),
                    "source": chunk
                }


    return best


# =========================================================
# 12. 最终问答
# =========================================================

def answer_question(query):

    companies = detect_companies(
        query
    )

    metric = detect_metric(
        query
    )


    # ----------------------------------
    # 标准财务指标
    # ----------------------------------

    if companies and metric:

        answers = []
        sources = []
        recall_records = []


        for company in companies:

            # 先做精确全表扫描
            evidence = find_metric_exact(
                company,
                metric
            )


            # 后台召回仍然保留，
            # 用于老师要求的实验记录
            company_results = search(
                query,
                company=company,
                top_k=5
            )


            for result in company_results:

                recall_records.append(
                    {
                        "company_target": company,
                        **result
                    }
                )


            if evidence is None:

                answers.append(
                    f"{company}："
                    f"未从财报表格中找到明确的"
                    f"{METRICS[metric]['name']}数据。"
                )

                continue


            answers.append(
                build_metric_answer(
                    company,
                    metric,
                    evidence
                )
            )


            sources.append(
                evidence["chunk"]
            )


        return {
            "answer": "\n".join(
                answers
            ),
            "sources": sources,
            "results": recall_records
        }


    # ----------------------------------
    # 非标准问题
    # ----------------------------------

    company_filter = None

    if len(companies) == 1:
        company_filter = companies[0]


    results = search(
        query,
        company=company_filter,
        top_k=5
    )


    generic = generic_answer(
        query,
        results
    )


    if generic is None:

        return {
            "answer":
                "暂未从财报中找到足够明确的答案。",
            "sources": [],
            "results": results
        }


    return {
        "answer": generic["answer"],
        "sources": [
            generic["source"]
        ],
        "results": results
    }


# =========================================================
# 13. 终端输出
# =========================================================

def main():

    query = input(
        "\n请输入财报问题："
    ).strip()


    if not query:

        raise ValueError(
            "问题不能为空。"
        )


    result = answer_question(
        query
    )


    print()
    print("=" * 70)
    print(f"问题：{query}")
    print("=" * 70)


    print()
    print("【回答】")

    print(
        result["answer"]
    )


    print()
    print("【出处】")


    if result["sources"]:

        seen = set()
        number = 1


        for source in result["sources"]:

            key = (
                source.get(
                    "source_file"
                ),
                source.get("page"),
                source.get("section")
            )


            if key in seen:
                continue


            seen.add(key)


            print(
                f"{number}. "
                f"{source.get('source_file', '')}，"
                f"第 {source.get('page', '')} 页，"
                f"{source.get('section', '')}"
            )


            number += 1


    else:

        print(
            "暂无明确出处。"
        )


    print()
    print("【后台召回记录】")


    for i, item in enumerate(
        result["results"],
        start=1
    ):

        chunk = item["chunk"]

        target = item.get(
            "company_target",
            chunk.get("company", "")
        )


        print(
            f"{i}. "
            f"{target} | "
            f"{chunk.get('company', '')} | "
            f"第 {chunk.get('page', '')} 页 | "
            f"{chunk.get('type', '')} | "
            f"得分 {item['score']:.4f}"
        )


if __name__ == "__main__":
    main()
