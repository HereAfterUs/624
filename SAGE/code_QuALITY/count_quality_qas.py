# count_quality_qas.py
import json
from pathlib import Path

DATA = Path("../dataset/QuALITY.v1.0.1.train")  # 改成你的真实路径

article_cnt = 0
qa_cnt = 0
with DATA.open(encoding="utf-8", errors="ignore") as f:
    rec = []
    depth = 0
    for line in f:
        rec.append(line)
        depth += line.count("{") - line.count("}")
        if depth == 0 and rec:  # 读完一个 JSON 记录
            obj = json.loads("".join(rec))
            rec = []
            article_cnt += 1
            qa_cnt += len(obj.get("questions", []))

print(f"Articles = {article_cnt:,}")
print(f"Questions = {qa_cnt:,}")
