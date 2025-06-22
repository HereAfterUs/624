import nltk
import re
import tiktoken
from pathlib import Path

from tqdm import tqdm

nltk.download('punkt', quiet=True)
enc = tiktoken.get_encoding("cl100k_base")

SRC_DIR = Path("../processed_narrativeqa")
DST_DIR = Path("../seg_dot_narrativeqa")
DST_DIR.mkdir(exist_ok=True)

MIN_CHARS = 60
SENT_SEP = re.compile(r'(?<!\.)\. (?=[A-Z])')  # 规则详见上条消息


def dot_segment(text: str, min_chars: int = MIN_CHARS):
    parts, chunks = SENT_SEP.split(text), []
    for sent in (p.strip() for p in parts if p.strip()):
        if chunks and len(sent) < min_chars:
            chunks[-1] += '. ' + sent
        else:
            chunks.append(sent)
    return chunks


for fp in tqdm(sorted(SRC_DIR.glob("*.txt")), desc="Segmenting"):
    out_path = DST_DIR / fp.name.replace(".txt", ".chunks.txt")
    if out_path.exists():
        tqdm.write(f"✓ skip {fp.name} (already done)")
        continue

    text = fp.read_text(encoding="utf-8", errors="ignore")
    chunks = dot_segment(text)

    with out_path.open("w", encoding="utf-8") as f:
        for s in chunks:
            f.write(s + "\n")  # 一行一句

    tqdm.write(f"→ {fp.stem}: {len(chunks)} chunks saved")
