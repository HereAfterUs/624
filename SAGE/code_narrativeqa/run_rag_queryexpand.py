"""
run_rag.py  –  Naive RAG + Module B (Progressive Semantic Drifting)
-------------------------------------------------------------------
1. 对原始问题生成 3 条递进式同义查询
2. 4 条查询各取 Top-5 → 去重合并
3. GPT-4o-mini 在合并后的 chunk 上生成答案
输出字段：document_id, question, prediction, answer1, answer2, chunk_ids
"""

import csv, json, faiss, requests
from pathlib import Path
import numpy as np
import pandas as pd
from openai import OpenAI
from tqdm import tqdm

# ============== 配置 ==============
EMB_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"
API_KEY = "sk-yRuc2168ce331878ed754bd7df810d634b2c3403705vf1a6"
BASE_URL = "https://api.gptsapi.net/v1"

EXP_TEMPERATURE = 0.3  # 生成扩展查询时的随机性
TOP_PER_QUERY = 5  # 每条查询检索 Top-k
INDEX_DIR = Path("../indexes_narrativeqa")
CSV_PATH = Path("../res_narrativeqa/sample_qas.csv")  # 300-QA 子集
OUT_CSV = Path("../res_narrativeqa/predictions_queryexpand_new.csv")

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
# ==================================

# -------- Prompt 模板（模块 B） --------
# QUERY_EXPAND_PROMPT = """You are an assistant for query expansion.
# Generate exactly three alternate queries using *Progressive Semantic Drifting*:
#
# • Variant 1: same meaning (rephrase only)
# • Variant 2: add one small, related variation
# • Variant 3: add another small, related variation
#
# Rules:
# 1. Keep the user's information need; do NOT add external facts.
# 2. Output only the list lines, numbered 1) 2) 3).
#
# Original Query: {user_query}
#
# OUTPUT
# 1) ...
# 2) ...
# 3) ...
# """
QUERY_EXPAND_PROMPT = """You are an assistant for query expansion.
Generate exactly three alternate queries that keep the **same meaning** but use different wording.

Guidelines
1. Preserve the user’s information need; do NOT add external facts or drift the topic.
2. Vary surface expression: synonyms, syntax change, or light re-ordering.
3. Output only the three numbered lines.

Original Query: {user_query}

OUTPUT
1) ...
2) ...
3) ...
"""


# -------- 向量化 --------
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


# -------- 读取故事索引 --------
def load_index(doc_id):
    idx = faiss.read_index(str(INDEX_DIR / f"{doc_id}.faiss"))
    chs = json.loads((INDEX_DIR / f"{doc_id}.chunks.json").read_text("utf-8"))
    return idx, chs


# -------- 模块 B：扩展查询生成 --------
def expand_query(orig_q: str) -> list[str]:
    prompt = QUERY_EXPAND_PROMPT.format(user_query=orig_q)
    rsp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=EXP_TEMPERATURE,
    )
    lines = rsp.choices[0].message.content.strip().splitlines()
    variants = []
    for line in lines:
        if line and line[0] in "123":
            variants.append(line.split(")", 1)[1].strip())
    # 若解析失败，用空字符串占位
    while len(variants) < 3:
        variants.append("")
    return [orig_q] + variants  # 共 4 条


# -------- 检索并去重 --------
def retrieve_chunks_multi(index, chunks, queries, k_each=TOP_PER_QUERY):
    embeds = embed(queries)  # (4, dim)
    id_score = {}  # chunk_id -> best score
    for q_vec in embeds:
        k = min(k_each, index.ntotal)
        D, I = index.search(q_vec.reshape(1, -1), k)
        for cid, score in zip(I[0], D[0]):
            if cid not in id_score or score > id_score[cid]:
                id_score[cid] = score
    sorted_ids = sorted(id_score, key=id_score.get, reverse=True)
    return sorted_ids, [chunks[i] for i in sorted_ids]


# -------- GPT 生成 --------
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


# -------- 主流程 --------
def main():
    df = pd.read_csv(CSV_PATH).sample(n=300, random_state=510)
    print("评测 300 QA • Naive RAG + B (Progressive Query Expansion)")

    rows = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        doc_id, q = row["document_id"], row["question"]
        index, chunk_texts = load_index(doc_id)

        queries = expand_query(q)  # 4 条查询
        id_list, ctx_chunks = retrieve_chunks_multi(
            index, chunk_texts, queries, TOP_PER_QUERY
        )

        pred = answer_llm(q, ctx_chunks)

        rows.append({
            "document_id": doc_id,
            "question": q,
            "prediction": pred,
            "answer1": row["answer1"],
            "answer2": row["answer2"],
            "chunk_ids": json.dumps([int(i) for i in id_list])
        })

    pd.DataFrame(rows).to_csv(OUT_CSV, index=False, quoting=csv.QUOTE_ALL)
    print(f"✓ 预测保存至 {OUT_CSV}")


if __name__ == "__main__":
    main()
