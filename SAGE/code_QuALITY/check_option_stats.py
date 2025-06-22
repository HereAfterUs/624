# check_option_stats.py
import json, collections
from pathlib import Path
from tqdm import tqdm

DATA_PATH = Path("../dataset/QuALITY.v1.0.1.train")


def iter_records(path):
    buf, depth = [], 0
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for ln in f:
            buf.append(ln)
            depth += ln.count("{") - ln.count("}")
            if depth == 0 and buf:
                yield json.loads("".join(buf))
                buf = []


dist = collections.Counter()
non_four = []  # (aid, q_idx, opt_len)

for rec in tqdm(iter_records(DATA_PATH), desc="scan"):
    aid = rec["article_id"]
    for i, q in enumerate(rec["questions"]):
        l = len(q["options"])
        dist[l] += 1
        if l != 4:
            non_four.append((aid, i, l))

print(f"\nOption length distribution: {dict(dist)}")

if non_four:
    print("---- NOT 4 options (len !=4) ----")
    for aid, idx, ln in non_four:
        print(f"article {aid}  q#{idx}  len={ln}")
else:
    print("All questions have exactly 4 options.")
