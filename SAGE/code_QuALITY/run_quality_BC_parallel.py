# run_quality_BC_parallel.py  ——  QuALITY-train 300  · 5-API ·  B+C
# ------------------------------------------------------------
import csv
import json
import math
import random
import re
import requests
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from openai import OpenAI
from tqdm import tqdm

# ---------- 路径 ----------
DATA_PATH = Path("../dataset/QuALITY.v1.0.1.train")
IDX_DIR = Path("../indexes_quality_dot")  # 规则切分后的向量库
OUT_CSV = Path("../res_quality/pred_all_new.csv")

# ---------- 全局参数 ----------
EMB_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"
TOP_K_CAND = 10  # 初始候选
GRAD_W = 3
THRESH = 0.5
MIN_K = 2

# ---------- 5 个 OpenAI Key ----------
KEYS = [
    "sk-EhB7f3f04f79da972d586db0ed8e54c46f647b3b704qZbGv",
    "sk-UFzf72834b608faa98b30fd5516cad26a833eab2f56doUCU",
    "sk-yRuc2168ce331878ed754bd7df810d634b2c3403705vf1a6",
    "sk-OAha68dab27aa3fedb8e4b5b7e6529c70a1d3f941a8nfOmC",
    "sk-9Ll07c38e4a483c5ecece690efa6e00f603e587b0f747DYI",
]
BASE_URL = "https://api.gptsapi.net/v1"
CLIENTS = [OpenAI(api_key=k, base_url=BASE_URL) for k in KEYS]


# ============================================================
# 1) 读取数据 —— 与 Naive RAG 完全一致
# ============================================================
def iter_json_records(path: Path):
    """逐条读取 JSONL，但是官方文件是“多行 JSON”，需要自己累加 {} 深度。"""
    buf, depth = [], 0
    with path.open(encoding="utf-8", errors="ignore") as f:
        for ln in f:
            buf.append(ln)
            depth += ln.count("{") - ln.count("}")
            if depth == 0 and buf:
                yield json.loads("".join(buf))
                buf = []


def load_first_300_questions():
    rows = []
    # ① 按顺序拿前 300 篇 article
    for rec in list(iter_json_records(DATA_PATH))[:300]:
        aid = rec["article_id"]
        # ② 把这一篇里的所有 questions 顺序加入
        for q in rec["questions"]:
            rows.append({
                "article_id": aid,
                "question": q["question"].strip(),
                "options": q["options"],
                "gold": q["gold_label"]  # 1~4
            })
    # ③ 如果超出 300 题，就截断（Naive 代码的做法）
    return rows[:300]


# ============================================================
# 2) 查询扩展（模块 B）
# ============================================================
PROMPT_QEXP = """You are an assistant for query rewriting.
Generate exactly three alternative phrasings that keep the same intent.

Original: {q}

1)
2)
3)
"""


def expand_queries(q, client):
    rsp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": PROMPT_QEXP.format(q=q)}],
        temperature=0.3,
    )
    lines = [ln.strip()[2:].strip() for ln in
             rsp.choices[0].message.content.splitlines()
             if ln[:2] in {"1)", "2)", "3)"}]
    return [q] + lines[:3]


# ============================================================
# 3) 动态 K（模块 C）
# ============================================================
def smooth_select(scores, w=GRAD_W, thr=THRESH, min_k=MIN_K):
    keep = list(range(min_k))
    for i in range(min_k, len(scores)):
        if scores[i] >= np.mean(scores[max(0, i - w):i]) * thr:
            keep.append(i)
        else:
            break
    return keep


# ============================================================
# 4) Faiss index 载入
# ============================================================
def load_index(aid: str):
    idx = faiss.read_index(str(IDX_DIR / f"{aid}.faiss"))
    ch = json.loads((IDX_DIR / f"{aid}.chunks.json").read_text("utf-8"))
    return idx, ch


# ------------------------------------------------------------
def embed(texts, client, retry=3, delay=2):
    for a in range(retry):
        try:
            rsp = requests.post(f"{BASE_URL}/embeddings",
                                headers={"Authorization": f"Bearer {client.api_key}",
                                         "Content-Type": "application/json"},
                                json={"input": texts, "model": EMB_MODEL,
                                      "encoding_format": "float"},
                                timeout=60)
            rsp.raise_for_status()
            vecs = np.asarray([d["embedding"] for d in rsp.json()["data"]],
                              dtype=np.float32)
            vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-10
            return vecs
        except requests.HTTPError:
            if a == retry - 1: raise
            time.sleep(delay * (a + 1))


# ============================================================
# 5) Prompt （与 Naive RAG 完全一致，回答 A/B/C/D）
# ============================================================
PROMPT_QA_SYS = ("You are a multiple-choice QA assistant. "
                 "Answer ONLY with a single letter (A/B/C/D).")


def build_prompt(q, opts, ctx):
    opts_txt = "\n".join(f"{chr(65 + i)}) {o}" for i, o in enumerate(opts))
    return [
        {"role": "system", "content": PROMPT_QA_SYS},
        {"role": "user",
         "content": f"<context>\n{ctx}\n</context>\n\n"
                    f"Question: {q}\n\nOptions:\n{opts_txt}\n\nAnswer:"}
    ]


def parse_letter(ans: str):
    m = re.search(r"[A-D]", ans.upper())
    return "ABCD".find(m.group()) if m else -1


# ============================================================
# 6) 单条 QA
# ============================================================
def run_one(task, per_key):
    idx, row = task
    cli = CLIENTS[idx // per_key]

    aid, q, opts, gold = row["article_id"], row["question"], row["options"], row["gold"]
    index, chunks = load_index(aid)

    # --- B: 扩展 & 检索 ---
    best = {}
    for qu in expand_queries(q, cli):
        qv = embed([qu], cli)[0]
        sc, ids = index.search(qv[None, :], TOP_K_CAND)
        for i, s in zip(ids[0], sc[0]):
            if i not in best or s > best[i]:
                best[i] = s
    pair = sorted(best.items(), key=lambda x: x[1], reverse=True)
    keep = smooth_select([p[1] for p in pair])
    ctx = "\n".join(chunks[pair[i][0]] for i in keep)

    # --- 生成答案 ---
    rsp = cli.chat.completions.create(
        model=LLM_MODEL,
        messages=build_prompt(q, opts, ctx),
        temperature=0.0,
    )
    pred = parse_letter(rsp.choices[0].message.content)
    return {"article_id": aid, "question": q,
            "prediction": pred, "gold": gold}


# ============================================================
# 7) 并行执行
# ============================================================
random.seed(510)
rows = load_first_300_questions()
random.shuffle(rows)  # 与 Naive 一致

PER_KEY = math.ceil(len(rows) / len(CLIENTS))  # 60
tasks = list(enumerate(rows))

results = []
with ThreadPoolExecutor(max_workers=len(CLIENTS)) as ex:
    futs = {ex.submit(run_one, t, PER_KEY): t[0] for t in tasks}
    for f in tqdm(as_completed(futs), total=len(futs)):
        results.append(f.result())

# ============================================================
# 8) 保存 & 打印准确率
# ============================================================
df = pd.DataFrame(results)
df.to_csv(OUT_CSV, index=False, quoting=csv.QUOTE_ALL)
acc = (df["prediction"] == df["gold"]).mean() * 100
print(f"\n✓ saved {OUT_CSV}")
print(f"Accuracy = {acc:.2f}%  on 300 questions")
