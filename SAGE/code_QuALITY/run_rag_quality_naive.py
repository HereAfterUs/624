# run_rag_quality_naive.py  ——  QuALITY-train 300  •  5-API 并行
import csv
import faiss
import json
import math
import numpy as np
import pandas as pd
import re
import requests
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm

from code_QuALITY.build_index_quality import enc

# ---------- 路径 ----------
DATA_PATH = Path("../dataset/QuALITY.v1.0.1.train")
IDX_DIR = Path("../indexes_quality")
OUT_CSV = Path("../res_quality/pred_quality_naive.csv")

# ---------- 参数 ----------
TOP_K = 5
EMB_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"

lock = threading.Lock()

KEYS = [
    "sk-EhB7f3f04f79da972d586db0ed8e54c46f647b3b704qZbGv",
    "sk-UFzf72834b608faa98b30fd5516cad26a833eab2f56doUCU",
    "sk-yRuc2168ce331878ed754bd7df810d634b2c3403705vf1a6",
    "sk-OAha68dab27aa3fedb8e4b5b7e6529c70a1d3f941a8nfOmC",
    "sk-9Ll07c38e4a483c5ecece690efa6e00f603e587b0f747DYI",
]
BASE_URL = "https://api.gptsapi.net/v1"
CLIENTS = [OpenAI(api_key=k, base_url=BASE_URL) for k in KEYS]


# ---------- 读取 QuALITY ----------
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


# ---------- 嵌入 ----------
def embed(texts, client):
    rsp = requests.post(
        f"{BASE_URL}/embeddings",
        headers={"Authorization": f"Bearer {client.api_key}",
                 "Content-Type": "application/json"},
        json={"input": texts, "model": EMB_MODEL, "encoding_format": "float"},
        timeout=60,
    )
    rsp.raise_for_status()
    vec = np.asarray([d["embedding"] for d in rsp.json()["data"]],
                     dtype=np.float32)
    vec /= np.linalg.norm(vec, axis=1, keepdims=True) + 1e-10
    return vec


# ---------- 索引缓存 ----------
CACHE = {}


def rebuild_one(aid: str):
    """读取原始 train 文件 → 切 200-token → 嵌入 → 写 .faiss / .chunks.json"""
    # --- 找原文 ---
    article = None
    with open(DATA_PATH, "r", encoding="utf-8", errors="ignore") as f:
        buf, depth = [], 0
        for line in f:
            buf.append(line)
            depth += line.count("{") - line.count("}")
            if depth == 0 and buf:
                rec = "".join(buf).strip();
                buf = []
                if rec:
                    obj = json.loads(rec)
                    if obj["article_id"] == aid:
                        article = obj["article"];
                        break
    if article is None:
        raise FileNotFoundError(f"article_id {aid} not in dataset")

    # --- 切分 & 嵌入 ---
    chks = []
    ids = enc.encode(article)
    for i in range(0, len(ids), 200):
        chks.append(enc.decode(ids[i:i + 200]))

    vecs = np.vstack([embed(chks[j:j + 64], CLIENTS[0])
                      for j in range(0, len(chks), 64)])
    idx = faiss.IndexFlatIP(vecs.shape[1]);
    idx.add(vecs)

    # --- 写回 ---
    faiss.write_index(idx, str(IDX_DIR / f"{aid}.faiss"))
    (IDX_DIR / f"{aid}.chunks.json").write_text(
        json.dumps(chks, ensure_ascii=False), encoding="utf-8")

    # 缓存
    CACHE[aid] = (idx, chks)
    return idx, chks


def load_index(aid: str):
    if aid in CACHE:
        return CACHE[aid]

    try:
        txt = (IDX_DIR / f"{aid}.chunks.json").read_text(encoding="utf-8")
        if not txt.strip():  # 空文件
            raise ValueError("empty")
        chks = json.loads(txt)
        idx = faiss.read_index(str(IDX_DIR / f"{aid}.faiss"))
        CACHE[aid] = (idx, chks)
        return idx, chks
    except (FileNotFoundError, ValueError, json.JSONDecodeError):
        # 抢锁，确保并发时只重建一次
        with lock:
            if aid in CACHE:  # 可能别的线程已经建完
                return CACHE[aid]
            return rebuild_one(aid)


# ---------- prompt 构造 ----------
def to_prompt(q, options, ctx):
    opts_txt = "\n".join(f"{chr(65 + i)}) {o}" for i, o in enumerate(options))
    return [
        {"role": "system",
         "content": "You are a multiple-choice QA assistant. "
                    "Answer ONLY with a single letter (A/B/C/D)."},
        {"role": "user",
         "content": f"<context>\n{ctx}\n</context>\n\n"
                    f"Question: {q}\n\nOptions:\n{opts_txt}\n\nAnswer:"}
    ]


# ---------- 单条 QA 处理 ----------
def run_one(idx: int, qrec: tuple, per_key: int):
    client = CLIENTS[idx // per_key]

    aid, qtxt, opts, gold = qrec
    index, chunks = load_index(aid)

    q_vec, = embed([qtxt], client)
    _, ids = index.search(q_vec[None, :], TOP_K)
    ctx = "\n".join(chunks[i] for i in ids[0])

    rsp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=to_prompt(qtxt, opts, ctx),
        temperature=0.0,
    )
    m = re.search(r"[A-D]", rsp.choices[0].message.content.upper())
    pred = "ABCD".find(m.group()) if m else -1

    return {"article_id": aid,
            "question": qtxt,
            "prediction": pred,
            "gold": gold}


# ---------- 构造任务列表 (共 300 Q) ----------
records = list(iter_quality_records(DATA_PATH))[:300]  # train 300
qa_rows = []
for rec in records:
    obj = json.loads(rec)
    aid = obj["article_id"]
    for q in obj["questions"]:
        qa_rows.append((aid,
                        q["question"].strip(),
                        q["options"],
                        q["gold_label"]))

# 保证正好 300 条
qa_rows = qa_rows[:300]
PER_KEY = math.ceil(len(qa_rows) / len(CLIENTS))  # 60

# ---------- 并行执行 ----------
results = []
with ThreadPoolExecutor(max_workers=len(CLIENTS)) as ex:
    futures = {ex.submit(run_one, i, t, PER_KEY): i
               for i, t in enumerate(qa_rows)}
    for fut in tqdm(as_completed(futures), total=len(futures)):
        results.append(fut.result())

# ---------- 保存 & 简评 ----------
df = pd.DataFrame(results)
df.to_csv(OUT_CSV, index=False, quoting=csv.QUOTE_ALL)
acc = (df["prediction"] == df["gold"]).mean() * 100
print(f"\n✓ saved {OUT_CSV}")
print(f"Accuracy = {acc:.2f}%  on 300 questions")
