# build_index_dot.py
"""
从 seg_dot_narrativeqa/*.chunks.txt 读取按句号规则切好的句子，
调用 OpenAI /embeddings 生成向量，写入 Faiss.IndexFlatIP
并把原始 chunk 文本保存到同名 .chunks.json。
"""

import faiss
import json
import numpy as np
import requests
from pathlib import Path

import tiktoken
from tqdm import tqdm

# -------- 全局配置 --------
TEXT_DIR = Path("../seg_dot_narrativeqa")  # 输入
OUT_DIR = Path("../indexes_dot_narrativeqa")  # 输出
OUT_DIR.mkdir(exist_ok=True)

EMB_MODEL = "text-embedding-3-small"
BATCH_SIZE = 64
API_KEY = "sk-yRuc2168ce331878ed754bd7df810d634b2c3403705vf1a6"  # ← 你的 key
BASE_URL = "https://api.gptsapi.net/v1"
# --------------------------

enc = tiktoken.get_encoding("cl100k_base")


def embed(texts):
    """调用 OpenAI /embeddings，返回 (n,d) 的 float32 numpy 数组（归一化）"""
    body = {"input": texts, "model": EMB_MODEL, "encoding_format": "float"}
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    rsp = requests.post(f"{BASE_URL}/embeddings", json=body, headers=headers, timeout=60)
    if rsp.status_code != 200:
        raise RuntimeError(f"Embedding error {rsp.status_code}: {rsp.text[:200]}")
    vecs = np.asarray([d["embedding"] for d in rsp.json()["data"]], dtype=np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-10
    return vecs


def process_one(txt_path: Path):
    doc_id = txt_path.stem.replace(".chunks", "")
    faiss_path = OUT_DIR / f"{doc_id}.faiss"
    chunks_path = OUT_DIR / f"{doc_id}.chunks.json"
    if faiss_path.exists() and chunks_path.exists():
        print(f"✓ skip {doc_id} (indexed)");
        return

    chunks = [ln.strip() for ln in txt_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    print(f"→ {doc_id}: {len(chunks)} chunks")

    vec_list = []
    for i in tqdm(range(0, len(chunks), BATCH_SIZE), desc=f"embed {doc_id}", leave=False):
        vec_list.append(embed(chunks[i:i + BATCH_SIZE]))
    all_vecs = np.vstack(vec_list)

    index = faiss.IndexFlatIP(all_vecs.shape[1])
    index.add(all_vecs)
    faiss.write_index(index, str(faiss_path))
    json.dump(chunks, chunks_path.open("w", encoding="utf-8"), ensure_ascii=False)
    print(f"✓ saved {doc_id}")


def main():
    txt_files = sorted(TEXT_DIR.glob("*.chunks.txt"))
    print(f"Need to build {len(txt_files)} indexes\n")
    for fp in txt_files:
        process_one(fp)


if __name__ == "__main__":
    main()
