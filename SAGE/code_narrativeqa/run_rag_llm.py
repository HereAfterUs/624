"""
Final evaluation: 300 QA  •  Top-5 retrieval
------------------------------------------------
读取 sample_qas.csv → 随机抽 300 条 → 检索-生成 →
保存到 predictions_top5_300.csv
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

# ---------- 参数 ----------
TOP_K = 10
QA_N = 300  # 抽样条数
RNG_SEED = 510
EMB_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"
API_KEY = "sk-yRuc2168ce331878ed754bd7df810d634b2c3403705vf1a6"
BASE_URL = "https://api.gptsapi.net/v1"

INDEX_DIR = Path("../indexes_llm_narrativeqa")
CSV_PATH = Path("../res_narrativeqa/sample_qas.csv")
OUT_CSV = Path("../res_narrativeqa/predictions_top10_300_llm.csv")

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)


# ---------- 嵌入 ----------
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


# ---------- 索引加载 ----------
def load_index(doc_id):
    idx = faiss.read_index(str(INDEX_DIR / f"{doc_id}.chunks.faiss"))
    chs = json.loads((INDEX_DIR / f"{doc_id}.chunks.chunks.json").read_text("utf-8"))
    return idx, chs


# ---------- 生成 ----------
def answer_llm(question, ctx_chunks):
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


# ---------- 主流程 ----------
def main():
    df = pd.read_csv(CSV_PATH).sample(n=QA_N, random_state=RNG_SEED)
    print(f"评测样本数：{len(df)} (随机种子={RNG_SEED})")

    outputs = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        doc_id, q = row["document_id"], row["question"]
        index, chunks = load_index(doc_id)
        q_vec = embed([q])
        _, idx = index.search(q_vec, TOP_K)
        ids = idx[0].tolist()
        ans = answer_llm(q, [chunks[i] for i in ids])

        outputs.append({
            "document_id": doc_id,
            "question": q,
            "prediction": ans,
            "answer1": row["answer1"],
            "answer2": row["answer2"],
            "chunk_ids": json.dumps(ids)
        })

    pd.DataFrame(outputs).to_csv(OUT_CSV, index=False, quoting=csv.QUOTE_ALL)
    print(f"✓ 结果已保存至 {OUT_CSV}")


if __name__ == "__main__":
    main()
