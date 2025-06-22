"""
RAG + Module-C: Smooth-Gradient Dynamic-K
----------------------------------------
1. 检索 Top-20 via Faiss
2. 用平滑梯度算法裁剪到动态 K (≈3-8)
3. GPT-4o-mini 生成答案
结果：document_id, question, prediction, answer1, answer2, chunk_ids
"""

import csv
import faiss
import json
import requests
from pathlib import Path

import numpy as np
import pandas as pd
from openai import OpenAI
from tqdm import tqdm

# ------------------ 参数区 ------------------
# TOP_CANDIDATE = 20  # 初步检索返回数
WIN_SIZE = 3  # Smooth window_size
MIN_K = 3  # 至少保留块
THRESH = 0.5  # 跌破阈值即截断

EMB_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"
API_KEY = "sk-yRuc2168ce331878ed754bd7df810d634b2c3403705vf1a6"
BASE_URL = "https://api.gptsapi.net/v1"

INDEX_DIR = Path("../indexes_narrativeqa")
CSV_PATH = Path("../res_narrativeqa/sample_qas.csv")
OUT_CSV = Path("predictions_dynK10_all.csv")

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)


# -------------------------------------------

def embed(texts):
    url = f"{BASE_URL}/embeddings"
    hdrs = {"Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}"}
    body = {"input": texts, "model": EMB_MODEL, "encoding_format": "float"}
    r = requests.post(url, headers=hdrs, json=body, timeout=60)
    r.raise_for_status()
    vecs = np.asarray([d["embedding"] for d in r.json()["data"]],
                      dtype=np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-10
    return vecs


def load_index(doc_id):
    idx = faiss.read_index(str(INDEX_DIR / f"{doc_id}.faiss"))
    chs = json.loads((INDEX_DIR / f"{doc_id}.chunks.json").read_text("utf-8"))
    return idx, chs


# --------- Module-C: Smooth-Gradient 剪裁 ----------
def smooth_gradient_select(chunks, scores,
                           window_size=WIN_SIZE,
                           min_k=MIN_K, threshold=THRESH):
    if len(chunks) <= min_k:  # 数据不足时直接返回全部
        return list(range(len(chunks)))

    selected = list(range(min_k))  # 保底前 min_k
    for i in range(min_k, len(chunks)):
        w_start = max(min_k - 1, i - window_size)
        window_avg = sum(scores[w_start:i]) / (i - w_start)
        if scores[i] >= window_avg * threshold:
            selected.append(i)
        else:
            break
    return selected  # 返回局部下标列表


# --------------------------------------------------

def answer_llm(question, ctx_chunks):
    context = "\n".join(ctx_chunks)
    sys = ("You are a factual QA assistant. "
           "Answer ONLY using <context>. If not present, say \"I don't know.\"")
    user = f"<context>\n{context}\n</context>\n\n<question>{question}</question>"
    rsp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "system", "content": sys},
                  {"role": "user", "content": user}],
        temperature=0.0,
    )
    return rsp.choices[0].message.content.strip()


def main():
    df = pd.read_csv(CSV_PATH).sample(n=300, random_state=510)
    print(f"评测 300 QA • Dynamic-K")

    outputs = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        doc_id, q = row["document_id"], row["question"]
        index, chunks = load_index(doc_id)

        q_vec = embed([q])
        # k = min(TOP_CANDIDATE, index.ntotal)  # <—— 新增
        k = index.ntotal
        D, I = index.search(q_vec, k)

        cand_scores = D[0].tolist()  # 高→低
        cand_chunks = [chunks[i] for i in I[0]]

        keep_local_idx = smooth_gradient_select(
            cand_chunks, cand_scores
        )
        keep_global_idx = [I[0][j] for j in keep_local_idx]
        ctx_chunks = [chunks[i] for i in keep_global_idx]

        pred = answer_llm(q, ctx_chunks)

        outputs.append({
            "document_id": doc_id,
            "question": q,
            "prediction": pred,
            "answer1": row["answer1"],
            "answer2": row["answer2"],
            "chunk_ids": json.dumps([int(i) for i in keep_global_idx])
        })

    pd.DataFrame(outputs).to_csv(OUT_CSV, index=False, quoting=csv.QUOTE_ALL)
    print(f"✓ 动态 K 预测写入 {OUT_CSV}")


if __name__ == "__main__":
    main()
