# build_index_quality_dot_retry.py
from pathlib import Path
import re, json, html, time, random, faiss, requests, numpy as np, tiktoken
from bs4 import BeautifulSoup
from tqdm import tqdm

DATA = Path("../dataset/QuALITY.v1.0.1.train")
OUT = Path("../indexes_quality_dot")
OUT.mkdir(exist_ok=True)

MIN_LEN = 40
CHUNK_TOK = 200
BATCH_SZ = 64  # ↓一点，减轻瞬时 QPS
EMB_MODEL = "text-embedding-3-small"
API_KEY = "sk-yRuc2168ce331878ed754bd7df810d634b2c3403705vf1a6"
BASE_URL = "https://api.gptsapi.net/v1"

enc = tiktoken.get_encoding("cl100k_base")
DOT_RE = re.compile(r"\.\s+")


# ---------- HTML → plain ----------
def clean_html(raw: str) -> str:
    txt = BeautifulSoup(raw, "lxml").get_text(" ")
    txt = html.unescape(txt)
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt


# ---------- dot split ----------
def dot_split(text: str):
    parts = []
    for sent in DOT_RE.split(text):
        sent = sent.strip()
        if not sent:
            continue
        if parts and len(parts[-1]) < MIN_LEN:
            parts[-1] += " " + sent
        else:
            parts.append(sent)
    return parts


# ---------- robust embed ----------
def embed(text_list, max_retry=5):
    for attempt in range(1, max_retry + 1):
        try:
            rsp = requests.post(
                f"{BASE_URL}/embeddings",
                headers={"Authorization": f"Bearer {API_KEY}",
                         "Content-Type": "application/json"},
                json={"input": text_list,
                      "model": EMB_MODEL,
                      "encoding_format": "float"},
                timeout=60,
            )
            rsp.raise_for_status()
            vecs = np.asarray([d["embedding"] for d in rsp.json()["data"]],
                              dtype=np.float32)
            vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-10
            return vecs
        except (requests.exceptions.RequestException, json.JSONDecodeError) as e:
            if attempt == max_retry:
                raise RuntimeError(f"embed failed after {max_retry} tries") from e
            wait = 2 ** attempt + random.random()
            print(f"⚠️  embed retry {attempt}/{max_retry} in {wait:.1f}s  ({e})")
            time.sleep(wait)


# ---------- JSONL iterator ----------
def iter_records(path):
    buf, depth = [], 0
    with open(path, encoding="utf-8", errors="ignore") as f:
        for ln in f:
            buf.append(ln)
            depth += ln.count("{") - ln.count("}")
            if depth == 0 and buf:
                yield json.loads("".join(buf));
                buf = []


# ---------- main ----------
for rec in tqdm(iter_records(DATA), desc="build"):
    aid = rec["article_id"]
    faiss_p = OUT / f"{aid}.faiss"
    chunk_p = OUT / f"{aid}.chunks.json"
    if faiss_p.exists():
        continue  # 已完成

    text = clean_html(rec["article"])
    chunks = dot_split(text)

    vecs_all = []
    for i in range(0, len(chunks), BATCH_SZ):
        vecs_all.append(embed(chunks[i:i + BATCH_SZ]))
        time.sleep(0.2)  # 轻量节流
    vecs = np.vstack(vecs_all)

    idx = faiss.IndexFlatIP(vecs.shape[1])
    idx.add(vecs)
    faiss.write_index(idx, str(faiss_p))
    chunk_p.write_text(json.dumps(chunks, ensure_ascii=False), encoding="utf-8")

print("✓ rule-based index (dot) completed")
