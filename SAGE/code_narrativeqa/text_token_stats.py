import csv
from pathlib import Path

import pandas as pd
import tiktoken

SAMPLE_CSV = Path("../res_narrativeqa/sample_qas.csv")
CONTENT_DIR = Path("../dataset/hg/narrativeqa/data/narrativeqa_full_text")
OUT_CSV = Path("../res_narrativeqa/text_token_stats_30.csv")

enc = tiktoken.get_encoding("cl100k_base")

doc_ids = pd.read_csv(SAMPLE_CSV)["document_id"].unique()
rows, total = [], 0
for doc_id in doc_ids:
    fp = CONTENT_DIR / f"{doc_id}.content"
    text = fp.read_text(encoding="utf-8", errors="ignore")
    n_tok = len(enc.encode(text))
    rows.append({"document_id": doc_id, "tokens": n_tok})
    total += n_tok

with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["document_id", "tokens"])
    writer.writeheader();
    writer.writerows(rows)

print(f"Written: {OUT_CSV} | 30-story total = {total:,} tokens")
