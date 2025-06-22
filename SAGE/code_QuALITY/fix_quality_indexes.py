# fix_quality_indexes.py  ——  rebuild non-UTF8 indexes  (修正版)

import json, faiss, requests, numpy as np, tiktoken
from pathlib import Path
from tqdm import tqdm

DATA_PATH = Path("../dataset/QuALITY.v1.0.1.train")
IDX_DIR = Path("../indexes_quality")

CHUNK_TOK = 200
API_KEY = "sk-yRuc2168ce331878ed754bd7df810d634b2c3403705vf1a6"  # ← 你的 key
BASE_URL = "https://api.gptsapi.net/v1"
EMB_MODEL = "text-embedding-3-small"

enc = tiktoken.get_encoding("cl100k_base")


# ---------- small helpers ----------
def embed(texts):
    rsp = requests.post(f"{BASE_URL}/embeddings",
                        headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
                        json={"input": texts, "model": EMB_MODEL, "encoding_format": "float"}, timeout=60)
    rsp.raise_for_status()
    v = np.asarray([d["embedding"] for d in rsp.json()["data"]], dtype=np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-10
    return v


def split_chunks(txt):
    ids = enc.encode(txt)
    for i in range(0, len(ids), CHUNK_TOK):
        yield enc.decode(ids[i:i + CHUNK_TOK])


def iter_records(path):
    buf, depth = [], 0
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            buf.append(line)
            depth += line.count("{") - line.count("}")
            if depth == 0 and buf:
                yield "".join(buf).strip();
                buf = []


# ---------- 1) detect bad json ----------
bad_ids = []
for fp in IDX_DIR.glob("*.chunks.json"):
    try:
        fp.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        aid = fp.stem.split(".")[0]  # <-- 修正：取数字部分
        bad_ids.append(aid)

print("Need rebuild:", bad_ids)
if not bad_ids:
    print("✓ already clean");
    exit()

# ---------- 2) fetch article texts ----------
text_map = {}
for rec in iter_records(DATA_PATH):
    obj = json.loads(rec)
    aid = obj["article_id"]
    if aid in bad_ids:
        text_map[aid] = obj["article"]
        if len(text_map) == len(bad_ids):
            break

# ---------- 3) rebuild ----------
for aid, article in tqdm(text_map.items(), desc="rebuild"):
    chunks = list(split_chunks(article))
    vecs = np.vstack([embed(chunks[i:i + 64]) for i in range(0, len(chunks), 64)])

    faiss_p = IDX_DIR / f"{aid}.faiss"
    chunk_p = IDX_DIR / f"{aid}.chunks.json"

    idx = faiss.IndexFlatIP(vecs.shape[1]);
    idx.add(vecs)
    faiss.write_index(idx, str(faiss_p))
    chunk_p.write_text(json.dumps(chunks, ensure_ascii=False), encoding="utf-8")

print("✓ rebuild finished — indexes are now all UTF-8")
