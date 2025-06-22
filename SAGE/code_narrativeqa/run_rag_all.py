# sys = "You are a factual QA assistant. Answer ONLY with information in <context>"

import csv
import json
import math
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import faiss
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ---------- SBERT encoder ----------
SBERT = SentenceTransformer("models/all-MiniLM-L6-v2", device="cuda")
SBERT.normalize_embeddings = True  # 内部 norm=True


def sbert_embed(texts, batch=64):
    vecs = SBERT.encode(texts, batch_size=batch, normalize_embeddings=True)
    return vecs.astype("float32")


# ---------- cache: doc_id -> (faissIndex, chunks) ----------
INDEX_CACHE = {}

# ---------- 多 API ----------
KEYS = [
    "sk-EhB7f3f04f79da972d586db0ed8e54c46f647b3b704qZbGv",
    "sk-UFzf72834b608faa98b30fd5516cad26a833eab2f56doUCU",
    "sk-yRuc2168ce331878ed754bd7df810d634b2c3403705vf1a6",
    "sk-OAha68dab27aa3fedb8e4b5b7e6529c70a1d3f941a8nfOmC",
    "sk-9Ll07c38e4a483c5ecece690efa6e00f603e587b0f747DYI",
]
BASE_URL = "https://api.gptsapi.net/v1"
CLIENTS = [OpenAI(api_key=k, base_url=BASE_URL) for k in KEYS]

# ---------- 参数 ----------
TOP_K_CAND = 10
WINDOW_W = 1
GRAD_W = 3
THRESH = 0.5
MIN_K = 2
LLM_MODEL = "gpt-4o-mini"

INDEX_DIR = Path("../indexes_dot_narrativeqa")  # 只需 .chunks.json
CSV_PATH = Path("../res_narrativeqa/sample_qas.csv")
OUT_CSV = Path("../res_narrativeqa/pred_sbert_all.csv")

# ---------- B 查询扩展 ----------
PROMPT = """You are an assistant for query rewriting.
Generate exactly three alternative phrasings that keep the same intent.

Original: {q}

1)
2)
3)
"""


def expand_queries(q, client):
    rsp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": PROMPT.format(q=q)}],
        temperature=0.3,
    )
    lines = [ln.strip()[2:].strip() for ln in
             rsp.choices[0].message.content.splitlines()
             if ln[:2] in {"1)", "2)", "3)"}]
    return [q] + lines[:3]


# ---------- C 动态截断 ----------
def smooth_select(scores, w=GRAD_W, thr=THRESH, min_k=MIN_K):
    keep = list(range(min_k))
    for i in range(min_k, len(scores)):
        win = scores[max(0, i - w):i]
        if scores[i] >= np.mean(win) * thr:
            keep.append(i)
        else:
            break
    return keep


# ---------- 工具 ----------
def win(chunks, idx, w=WINDOW_W):
    l = max(0, idx - w)
    r = min(len(chunks), idx + w + 1)
    return " ".join(chunks[l:r])


def load_index(doc_id):
    """若缓存中无，现场构建 SBERT 向量 + Faiss."""
    if doc_id in INDEX_CACHE:
        return INDEX_CACHE[doc_id]

    chunks = json.loads((INDEX_DIR / f"{doc_id}.chunks.json")
                        .read_text(encoding="utf-8", errors="ignore"))
    vecs = sbert_embed(chunks, batch=128)
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)
    INDEX_CACHE[doc_id] = (index, chunks)
    return index, chunks


def gpt_answer(q, ctx, client):
    sys = "You are a factual QA assistant. Answer ONLY with information in <context>"
    user = f"<context>\n{ctx}\n</context>\n\n<question>{q}</question>"
    rsp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "system", "content": sys},
                  {"role": "user", "content": user}],
        temperature=0.0,
    )
    return rsp.choices[0].message.content.strip()


# ---------- 单 QA ----------
def run_one(task, per_key):
    idx, row = task
    client = CLIENTS[idx // per_key]

    doc_id, q = row["document_id"], row["question"]
    index, chunks = load_index(doc_id)

    queries = expand_queries(q, client)

    best = {}
    for qu in queries:
        qv = sbert_embed([qu])
        sc, ids = index.search(qv, TOP_K_CAND)
        for i, s in zip(ids[0], sc[0]):
            if i not in best or s > best[i]:
                best[i] = s

    pair = sorted(best.items(), key=lambda x: x[1], reverse=True)
    ids_sorted = [p[0] for p in pair]
    keep = smooth_select([p[1] for p in pair])

    ctx = "\n".join(win(chunks, ids_sorted[i]) for i in keep)
    pred = gpt_answer(q, ctx, client)

    return {
        "document_id": doc_id,
        "question": q,
        "prediction": pred,
        "answer1": row["answer1"],
        "answer2": row["answer2"],
        "chunk_ids": json.dumps([int(i) for i in ids_sorted[:len(keep)]])
    }


# ---------- 主 ----------
df = (pd.read_csv(CSV_PATH)
      .sample(n=300, random_state=510)
      .reset_index(drop=True))
tasks = list(df.iterrows())
PER_KEY = math.ceil(len(tasks) / len(CLIENTS))  # 60

results = []
with ThreadPoolExecutor(max_workers=len(CLIENTS)) as ex:
    futures = {ex.submit(run_one, t, PER_KEY): t[0] for t in tasks}
    for fut in tqdm(as_completed(futures), total=len(tasks)):
        results.append(fut.result())

pd.DataFrame(results).to_csv(OUT_CSV, index=False, quoting=csv.QUOTE_ALL)
print(f"✓ saved {OUT_CSV}")
