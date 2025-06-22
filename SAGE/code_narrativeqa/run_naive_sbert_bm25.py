import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from openai import OpenAI
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

TOP_K = 5
QA_N = 300
SEED = 510
LLM_MODEL = "gpt-4o-mini"

API_KEY = "sk-yRuc2168ce331878ed754bd7df810d634b2c3403705vf1a6"
BASE_URL = "https://api.gptsapi.net/v1"
INDEX_DIR = Path("../indexes_narrativeqa")
CSV_PATH = Path("../res_narrativeqa/sample_qas.csv")
OUT_DIR = Path("../res_narrativeqa")
OUT_DIR.mkdir(exist_ok=True)

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)


# ---------- 生成函数 ----------
def gpt_answer(question: str, ctx_chunks: list[str]) -> str:
    context = "\n".join(ctx_chunks)
    sys = ("You are a factual QA assistant. Answer ONLY with information in "
           "<context>. If the answer is not present, reply \"I don't know.\"")
    user = f"<context>\n{context}\n</context>\n\n<question>{question}</question>"
    rsp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "system", "content": sys},
                  {"role": "user", "content": user}],
        temperature=0.0,
    )
    return rsp.choices[0].message.content.strip()


# ========== SBERT 准备 ==========
print("🔹 loading SBERT encoder …")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

sbert = SentenceTransformer("../models/all-MiniLM-L6-v2", device=DEVICE)
print("SBERT device:", DEVICE)


def rank_sbert(question, chunk_texts, chunk_vecs):
    q = sbert.encode(question, normalize_embeddings=True)
    sims = np.dot(chunk_vecs, q)
    return np.argsort(sims)[::-1][:TOP_K]


# ========== 主循环 ==========
for mode in ("sbert", "bm25"):
    print(f"\n=== Running {mode.upper()} with NaiveRAG (Top-{TOP_K}) ===")
    outfile = OUT_DIR / f"pred_{mode}_top{TOP_K}.csv"
    rows = []

    df = pd.read_csv(CSV_PATH).sample(n=QA_N, random_state=SEED)

    # 缓存 story 索引 → 避免重复计算
    story_cache = {}  # doc_id : (chunk_texts, extra_obj)

    for _, row in tqdm(df.iterrows(), total=len(df)):
        doc_id, ques = row["document_id"], row["question"]

        if doc_id not in story_cache:
            chunks = json.loads((INDEX_DIR / f"{doc_id}.chunks.json").read_text())
            if mode == "sbert":
                vecs = sbert.encode(chunks, normalize_embeddings=True, batch_size=256)
                story_cache[doc_id] = (chunks, vecs)  # tuple
            else:  # bm25
                toks = [c.lower().split() for c in chunks]
                bm25 = BM25Okapi(toks)
                story_cache[doc_id] = (chunks, bm25)

        chunks, extra = story_cache[doc_id]

        # ------- 检索 -------
        if mode == "sbert":
            top_idx = rank_sbert(ques, chunks, extra)
        else:  # bm25
            scores = extra.get_scores(ques.lower().split())
            top_idx = np.argsort(scores)[::-1][:TOP_K]

        ctx = [chunks[i] for i in top_idx]
        pred = gpt_answer(ques, ctx)

        rows.append({
            "document_id": doc_id,
            "question": ques,
            "prediction": pred,
            "answer1": row["answer1"],
            "answer2": row["answer2"],
            "chunk_ids": json.dumps(top_idx.tolist()),
        })

    pd.DataFrame(rows).to_csv(outfile, index=False, quoting=csv.QUOTE_ALL)
    print(f"✓ saved {outfile.name}")

print("\nAll done — use eval_naive.py on the two CSVs.")
