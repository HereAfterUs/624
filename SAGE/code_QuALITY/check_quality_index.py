# check_quality_index.py
import json, glob
from pathlib import Path

DATA_PATH = Path("../dataset/QuALITY.v1.0.1.train")  # ← 路径按需改
INDEX_DIR = Path("../indexes_quality")  # ← 索引目录


def iter_quality_records(path):
    buf, depth = [], 0
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            buf.append(line)
            depth += line.count("{") - line.count("}")
            if depth == 0 and buf:
                rec = "".join(buf).strip()
                if rec:
                    yield rec
                buf = []


# ---------- 统计数据集 ----------
article_ids = set()
sample_cnt = 0
for rec in iter_quality_records(DATA_PATH):
    obj = json.loads(rec)
    article_ids.add(obj["article_id"])
    sample_cnt += 1

# ---------- 统计索引文件 ----------
faiss_files = glob.glob(str(INDEX_DIR / "*.faiss"))
index_cnt = len(faiss_files)

# ---------- 输出 ----------
print("=== QuALITY 统计 ===")
print(f"JSON records         : {sample_cnt}")
print(f"Unique article_ids   : {len(article_ids)}")
print(f".faiss index files   : {index_cnt}")

if len(article_ids) == index_cnt:
    print("✅  Index count matches unique article_ids.")
else:
    diff = len(article_ids) - index_cnt
    print(f"⚠️  Mismatch!  Difference = {diff} "
          f"({'missing indexes' if diff > 0 else 'extra files'}).")
