"""
build_index_llm.py
------------------
读取 segmented_narrativeqa/*.jsonl  (LLM 语义分割结果)
→ 去重 → 批量嵌入 → Faiss 索引
输出至 indexes_llm_narrativeqa/
"""

import json, requests, numpy as np, faiss
from pathlib import Path
from tqdm import tqdm

# -------- 参数 --------
SEG_DIR = Path("../segmented_narrativeqa")  # 输入 .jsonl
OUT_DIR = Path("../indexes_llm_narrativeqa")  # 输出索引
OUT_DIR.mkdir(exist_ok=True)

EMB_MODEL = "text-embedding-3-small"
BATCH_SIZE = 64

API_KEY = "sk-yRuc2168ce331878ed754bd7df810d634b2c3403705vf1a6"
BASE_URL = "https://api.gptsapi.net/v1"


# ----------------------

def embed(texts: list[str]) -> np.ndarray:
    url = f"{BASE_URL}/embeddings"
    hdrs = {"Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}"}
    body = {"input": texts, "model": EMB_MODEL, "encoding_format": "float"}
    rsp = requests.post(url, headers=hdrs, json=body, timeout=60)
    rsp.raise_for_status()
    vecs = np.asarray([d["embedding"] for d in rsp.json()["data"]],
                      dtype=np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-10
    return vecs


def process_file(jl_path: Path):
    doc_id = jl_path.stem  # e.g. "00fb61..."
    faiss_p = OUT_DIR / f"{doc_id}.faiss"
    chunk_p = OUT_DIR / f"{doc_id}.chunks.json"
    if faiss_p.exists() and chunk_p.exists():
        print(f"✓ skip {doc_id} (already built)")
        return

    # -------- 收集 & 去重 --------
    unique_chunks = []
    seen = set()
    with jl_path.open(encoding="utf-8") as f:
        for line in f:
            chunks = json.loads(line)["chunks"]
            for ck in chunks:
                if ck not in seen:
                    seen.add(ck)
                    unique_chunks.append(ck)

    print(f"→ {doc_id}: {len(unique_chunks)} unique chunks")

    # -------- 嵌入 --------
    vec_list = []
    for i in tqdm(range(0, len(unique_chunks), BATCH_SIZE),
                  desc=f"embed {doc_id}"):
        vec_list.append(embed(unique_chunks[i:i + BATCH_SIZE]))
    all_vecs = np.vstack(vec_list)

    # -------- 建 Faiss --------
    dim = all_vecs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(all_vecs)
    faiss.write_index(index, str(faiss_p))

    chunk_p.write_text(json.dumps(unique_chunks, ensure_ascii=False),
                       encoding="utf-8")
    print(f"✓ saved index for {doc_id}")


def main():
    for jl in sorted(SEG_DIR.glob("*.jsonl")):
        process_file(jl)


if __name__ == "__main__":
    main()
