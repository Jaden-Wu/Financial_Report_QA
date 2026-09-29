from pathlib import Path
import json

BASE_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = BASE_DIR / "processed"

OUTPUT_FILE = PROCESSED_DIR / "chunks.jsonl"

CHUNK_SIZE = 800
OVERLAP = 150


def split_text(text, chunk_size=CHUNK_SIZE, overlap=OVERLAP):
    text = text.strip()

    if not text:
        return []

    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = end - overlap

    return chunks


jsonl_files = [
    p for p in PROCESSED_DIR.glob("*.jsonl")
    if p.name != "chunks.jsonl"
]

if not jsonl_files:
    raise FileNotFoundError("processed 文件夹中没有找到财报解析结果。")


chunk_count = 0

with OUTPUT_FILE.open("w", encoding="utf-8") as output:

    for file_path in sorted(jsonl_files):

        print(f"正在切块：{file_path.name}")

        with file_path.open("r", encoding="utf-8") as f:

            for line in f:

                record = json.loads(line)

                company = record["company"]
                source_file = record["source_file"]
                page = record["page"]
                section = record["section"]
                text = record.get("text", "")
                tables = record.get("tables", [])

                # 正文切块
                text_chunks = split_text(text)

                for chunk_text in text_chunks:

                    chunk_count += 1

                    chunk_record = {
                        "chunk_id": chunk_count,
                        "company": company,
                        "source_file": source_file,
                        "page": page,
                        "section": section,
                        "type": "text",
                        "content": chunk_text
                    }

                    output.write(
                        json.dumps(
                            chunk_record,
                            ensure_ascii=False
                        ) + "\n"
                    )

                # 表格单独作为 chunk
                for table_number, table_text in enumerate(tables, start=1):

                    if not table_text.strip():
                        continue

                    chunk_count += 1

                    chunk_record = {
                        "chunk_id": chunk_count,
                        "company": company,
                        "source_file": source_file,
                        "page": page,
                        "section": section,
                        "type": "table",
                        "table_number": table_number,
                        "content": table_text
                    }

                    output.write(
                        json.dumps(
                            chunk_record,
                            ensure_ascii=False
                        ) + "\n"
                    )


print()
print("全部财报切块完成。")
print(f"共生成 {chunk_count} 个 chunk。")
print(f"结果保存到：{OUTPUT_FILE}")