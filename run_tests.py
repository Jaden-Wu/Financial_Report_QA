from pathlib import Path
import json
import subprocess
import sys
import os
from datetime import datetime


BASE_DIR = Path(__file__).resolve().parent

QUESTIONS_FILE = BASE_DIR / "test_questions.json"
QA_ENGINE = BASE_DIR / "qa_engine.py"

RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

OUTPUT_FILE = RESULTS_DIR / "test_results.txt"


# =========================
# 1. 读取测试题
# =========================

with QUESTIONS_FILE.open(
    "r",
    encoding="utf-8"
) as f:

    questions = json.load(f)


print(f"共读取 {len(questions)} 道测试题。")


# =========================
# 2. 为子进程强制使用 UTF-8
# =========================

child_env = os.environ.copy()

child_env["PYTHONIOENCODING"] = "utf-8"

# 兼容你当前 Python 环境
child_env["PYTHONPATH"] = r"D:\Lib\site-packages"


# =========================
# 3. 逐题运行 qa_engine.py
# =========================

all_results = []


for item in questions:

    qid = item["id"]
    qtype = item["type"]
    question = item["question"]

    print()
    print("=" * 60)
    print(
        f"正在测试第 {qid} 题：{question}"
    )
    print("=" * 60)

    process = subprocess.run(
        [
            sys.executable,
            str(QA_ENGINE)
        ],
        input=question + "\n",
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="strict",
        env=child_env
    )

    output = process.stdout

    # 只有程序真正失败时，
    # 才把 stderr 当作错误记录
    error_text = ""

    if process.returncode != 0:

        error_text = process.stderr


    all_results.append(
        {
            "id": qid,
            "type": qtype,
            "question": question,
            "output": output,
            "error": error_text,
            "returncode": process.returncode
        }
    )


    if process.returncode == 0:

        print("运行完成。")

    else:

        print("本题运行失败。")


# =========================
# 4. 保存测试结果
# =========================

with OUTPUT_FILE.open(
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "财报问答知识库测试记录\n"
    )

    f.write(
        "生成时间："
        + datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        + "\n"
    )

    f.write(
        f"测试题数量：{len(all_results)}\n"
    )


    for result in all_results:

        f.write("\n")
        f.write("=" * 80 + "\n")

        f.write(
            f"题号：{result['id']}\n"
        )

        f.write(
            f"类型：{result['type']}\n"
        )

        f.write(
            f"问题：{result['question']}\n"
        )

        f.write(
            f"程序返回码："
            f"{result['returncode']}\n"
        )

        f.write("-" * 80 + "\n")

        f.write(
            result["output"]
        )


        if result["error"]:

            f.write("\n")
            f.write("【程序错误】\n")

            f.write(
                result["error"]
            )


        f.write("\n")
        f.write("【人工评估】\n")
        f.write(
            "回答是否正确：待填写\n"
        )

        f.write(
            "错误或不足：待填写\n"
        )


print()
print("=" * 60)

print(
    "10道题已全部测试完成。"
)

print(
    f"测试记录已保存到："
    f"{OUTPUT_FILE}"
)

print("=" * 60)