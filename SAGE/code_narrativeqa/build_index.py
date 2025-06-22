"""
构建向量数据库
----------------------------------
1. 遍历 narrativeqa_full_text/*.content 读取全文
2. 固定 200 token 切分
3. 调用 /embeddings 获得向量并归一化
4. 建立 Faiss.IndexFlatIP，保存 .faiss 及对应 chunks.json
"""

import json
import time
from pathlib import Path
import random
import faiss
import numpy as np
import pandas as pd
import requests
import tiktoken
from tqdm import tqdm

# 全局参数
CHUNK_TOKENS = 200
EMB_MODEL = "text-embedding-3-small"
BATCH_SIZE = 64

STORY_NUM = 30
RNG_SEED = 510

random.seed(RNG_SEED)
np.random.seed(RNG_SEED)

API_KEY = "sk-yRuc2168ce331878ed754bd7df810d634b2c3403705vf1a6"
BASE_URL = "https://api.gptsapi.net/v1"

ENC = tiktoken.get_encoding("cl100k_base")
CONTENT_DIR = Path("../dataset/hg/narrativeqa/data/narrativeqa_full_text")
OUT_DIR = Path("../indexes_narrativeqa")
OUT_DIR.mkdir(exist_ok=True)


def split_to_chunks(text: str, max_tokens: int = CHUNK_TOKENS):
    """固定 token 切分"""
    ids = ENC.encode(text)
    for i in range(0, len(ids), max_tokens):
        yield ENC.decode(ids[i:i + max_tokens])


def embed(texts, max_retry=6):
    """调用 /embeddings，返回 (n, d) 的 float32 numpy 数组。"""
    url = f"{BASE_URL}/embeddings"
    hdrs = {"Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}"}
    body = {"input": texts, "model": EMB_MODEL, "encoding_format": "float"}
    for attempt in range(max_retry):
        rsp = requests.post(url, headers=hdrs, json=body, timeout=60)

        if rsp.status_code == 200:
            vecs = [d["embedding"] for d in rsp.json()["data"]]
            arr = np.asarray(vecs, dtype=np.float32)
            arr /= np.linalg.norm(arr, axis=1, keepdims=True) + 1e-10
            return arr

        # ---------- 429 处理 ----------
        if rsp.status_code == 429:
            wait = 5 * (2 ** attempt)  # 5s, 10s, 20s, 40s, ...
            print(f"⚠ 429 RateLimit，第 {attempt + 1}/{max_retry} 次退避 {wait}s")
            time.sleep(wait)
            continue

        # ---------- 其他错误 ----------
        raise RuntimeError(f"嵌入失败: {rsp.status_code} {rsp.text[:200]}")

        # 超过 max_retry 次仍失败
    raise RuntimeError("多次重试仍 429，终止脚本")


def process_one(content_path: Path):
    doc_id = content_path.stem
    faiss_path = OUT_DIR / f"{doc_id}.faiss"
    chunks_path = OUT_DIR / f"{doc_id}.chunks.json"

    # 若已存在索引则跳过
    if faiss_path.exists() and chunks_path.exists():
        print(f"✓ 已存在索引 跳过 {doc_id}\n")
        return

    text = content_path.read_text(encoding="utf-8", errors="ignore")
    chunks = list(split_to_chunks(text))
    print(f"→ {doc_id} 共有 {len(chunks)} 个块\n")

    # 批量嵌入
    vec_list = []
    for i in tqdm(range(0, len(chunks), BATCH_SIZE), desc=f"嵌入 {doc_id}\n"):
        vec_list.append(embed(chunks[i:i + BATCH_SIZE]))
    all_vecs = np.vstack(vec_list)

    # 构建 Faiss
    dim = all_vecs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(all_vecs)
    faiss.write_index(index, str(faiss_path))

    # 保存块文本
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False)

    print(f"✓ 已保存 {doc_id} 索引\n")


def main():
    df = pd.read_csv("../dataset/hg/narrativeqa/data/narrativeqa-master/narrativeqa-master/qaps.csv")
    valid_test = df[df["set"].isin(["valid", "test"])]

    all_ids = valid_test["document_id"].unique()
    chosen_ids = np.random.choice(all_ids, size=STORY_NUM, replace=False)

    sample_rows = valid_test[valid_test["document_id"].isin(chosen_ids)]
    sample_rows.to_csv("sample_qas.csv", index=False)
    print(f"已选 {STORY_NUM} 篇故事，共 {len(sample_rows)} 条 QA，写入 sample_qas.csv\n")

    files = [CONTENT_DIR / f"{doc_id}.content" for doc_id in chosen_ids]
    print(f"准备构建 {len(files)} 个索引\n")

    for fp in files:
        if fp.exists():
            process_one(fp)
        else:
            print(f"⚠ 找不到正文文件 跳过 {fp.name}")


if __name__ == "__main__":
    main()
