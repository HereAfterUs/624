import json
from pathlib import Path

INDEX_DIR = Path("../indexes_narrativeqa")
OUT_DIR = Path("../texts_narrativeqa")
OUT_DIR.mkdir(exist_ok=True)

for fp in INDEX_DIR.glob("*.chunks.json"):
    doc_id = fp.stem  # 去掉后缀 .chunks.json
    chunks = json.loads(fp.read_text(encoding="utf-8"))
    full_text = "".join(chunks)  # ❶ 不加任何分隔符

    out_path = OUT_DIR / f"{doc_id}.txt"
    out_path.write_text(full_text, encoding="utf-8")
    print(f"✓ restored {doc_id} -> texts_narrativeqa/{doc_id}.txt")
