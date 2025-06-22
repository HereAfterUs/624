# build_index_quality.py  ——  构建向量库 (200-token 固定切分)

import json, faiss, requests, numpy as np, tiktoken
from pathlib import Path
from tqdm import tqdm

# ---------- 路径 ----------
DATA_PATH = Path("../dataset/QuALITY.v1.0.1.train")  # 官方 train 文件
OUT_DIR = Path("../indexes_quality")
OUT_DIR.mkdir(exist_ok=True)

# ---------- 参数 ----------
CHUNK_TOK = 200
EMB_MODEL = "text-embedding-3-small"
API_KEY = "sk-yRuc2168ce331878ed754bd7df810d634b2c3403705vf1a6"
BASE_URL = "https://api.gptsapi.net/v1"

enc = tiktoken.get_encoding("cl100k_base")


# ---------- 嵌入 ----------
def embed(texts):
    rsp = requests.post(
        f"{BASE_URL}/embeddings",
        headers={"Authorization": f"Bearer {API_KEY}",
                 "Content-Type": "application/json"},
        json={"input": texts, "model": EMB_MODEL, "encoding_format": "float"},
        timeout=60,
    )
    rsp.raise_for_status()
    vecs = np.asarray([d["embedding"] for d in rsp.json()["data"]],
                      dtype=np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-10
    return vecs


# ---------- 200-token 切分 ----------
def split_chunks(text):
    ids = enc.encode(text)
    for i in range(0, len(ids), CHUNK_TOK):
        yield enc.decode(ids[i:i + CHUNK_TOK])


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


# ---------- 读取文件：空行分段 ----------
raw = DATA_PATH.read_text(encoding="utf-8", errors="ignore")
records = raw.strip().split("\n\n")  # 每段 = 完整 JSON

# -------------- 主循环 --------------
for rec in tqdm(iter_quality_records(DATA_PATH), desc="build"):
    obj = json.loads(rec)
    art_id = obj["article_id"]
    faiss_p = OUT_DIR / f"{art_id}.faiss"
    chunk_p = OUT_DIR / f"{art_id}.chunks.json"
    if faiss_p.exists():
        continue  # 同 ID 已处理

    chunks = list(split_chunks(obj["article"]))

    # 嵌入 & 写索引
    vecs = np.vstack([embed(chunks[i:i + 64])
                      for i in range(0, len(chunks), 64)])
    idx = faiss.IndexFlatIP(vecs.shape[1]);
    idx.add(vecs)
    faiss.write_index(idx, str(faiss_p))
    chunk_p.write_text(json.dumps(chunks, ensure_ascii=False),
                       encoding="utf-8")

print("✓ all articles indexed →", OUT_DIR)
