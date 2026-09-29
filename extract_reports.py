from pathlib import Path
import json
import re

try:
    import pdfplumber
except ImportError:
    raise ImportError("缺少 pdfplumber，下一步我们再安装。")


# =========================
# 1. 项目路径
# =========================

BASE_DIR = Path(__file__).resolve().parent
REPORTS_DIR = BASE_DIR / "reports"
PROCESSED_DIR = BASE_DIR / "processed"

PROCESSED_DIR.mkdir(exist_ok=True)


# =========================
# 2. 简单识别章节标题
# =========================

def detect_section(text, current_section):
    """
    从页面文字中尝试识别章节标题。
    如果本页没有明显标题，则沿用上一页章节。
    """

    if not text:
        return current_section

    lines = [line.strip() for line in text.splitlines() if line.strip()]

    patterns = [
        r"^第[一二三四五六七八九十百\d]+[章节]",
        r"^[一二三四五六七八九十]+、",
    ]

    for line in lines[:15]:
        for pattern in patterns:
            if re.match(pattern, line):
                return line[:80]

    return current_section


# =========================
# 3. 表格转成可检索文字
# =========================

def table_to_text(table):
    """
    将 PDF 中识别出的表格按行列恢复为文字。
    每一行使用 | 分隔。
    """

    rows = []

    for row in table:
        if not row:
            continue

        clean_row = []

        for cell in row:
            if cell is None:
                cell = ""

            cell = str(cell).replace("\n", " ").strip()
            clean_row.append(cell)

        rows.append(" | ".join(clean_row))

    return "\n".join(rows)


# =========================
# 4. 逐份解析财报
# =========================

pdf_files = sorted(REPORTS_DIR.glob("*.pdf"))

if not pdf_files:
    raise FileNotFoundError("reports 文件夹中没有找到 PDF。")

print(f"共发现 {len(pdf_files)} 份财报。")


for pdf_path in pdf_files:

    # 例如：比亚迪25.pdf → 比亚迪
    company = re.sub(r"25$", "", pdf_path.stem)

    print(f"\n正在处理：{company}")

    output_path = PROCESSED_DIR / f"{company}.jsonl"

    current_section = "未识别章节"

    with pdfplumber.open(pdf_path) as pdf, \
            output_path.open("w", encoding="utf-8") as f:

        total_pages = len(pdf.pages)

        for page_number, page in enumerate(pdf.pages, start=1):

            # 提取正文
            text = page.extract_text() or ""

            # 判断章节
            current_section = detect_section(
                text,
                current_section
            )

            # 提取表格
            tables = page.extract_tables()

            table_texts = []

            for table in tables:
                table_text = table_to_text(table)

                if table_text.strip():
                    table_texts.append(table_text)

            # 每一页保存一条记录
            record = {
                "company": company,
                "source_file": pdf_path.name,
                "page": page_number,
                "section": current_section,
                "text": text,
                "tables": table_texts
            }

            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False
                ) + "\n"
            )

            print(
                f"\r  已处理 {page_number}/{total_pages} 页",
                end=""
            )

    print(f"\n  已保存：{output_path.name}")


print("\n全部财报解析完成。")