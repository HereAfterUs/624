# STORY_PATH = Path(
#     "./dataset/hg/narrativeqa/data/narrativeqa_full_text/"
#     "0a93e857113efca05c6274e7af3ba4f03a023b9f.content"
# )
# API_KEY = "sk-yRuc2168ce331878ed754bd7df810d634b2c3403705vf1a6"


import json, time, tiktoken, re
from pathlib import Path
from openai import OpenAI

# --------- 配置 ---------
API_KEY = "sk-yRuc2168ce331878ed754bd7df810d634b2c3403705vf1a6"  # 你的 Key
BASE_URL = "https://api.gptsapi.net/v1"
SRC_DIR = Path("../processed_narrativeqa")
OUT_DIR = Path("../segmented_narrativeqa")
OUT_DIR.mkdir(exist_ok=True)
MAX_CHARS = 10000
PROMPT_TMPL = """
You are a segmentation assistant.
Split the following passage into coherent semantic chunks (~150-300 GPT tokens).

Output format (exactly):
### The exact text of chunk 1
### The exact text of chunk 2
(no extra lines, after each "### " write the original text of that chunk)

PASSAGE
<<<
{passage}
>>>

"""
# ------------------------

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
enc = tiktoken.get_encoding("cl100k_base")  # 仅用于统计


def next_slice(text: str, start: int) -> tuple[str, int]:
    """从 start 取 ≤MAX_CHARS，回溯到最近 '.'"""
    end = min(start + MAX_CHARS, len(text))
    slice_ = text[start:end]
    if end < len(text):  # 不是最后一段
        dot = slice_.rfind(".")
        if dot == -1:
            dot = len(slice_)  # 实在没有就硬切
        end = start + dot + 1  # 包含 '.'
        slice_ = text[start:end]
    return slice_, end


def call_llm(passage: str, retry=3):
    prompt = PROMPT_TMPL.format(passage=passage)
    for i in range(retry):
        try:
            rsp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                timeout=120,
            )
            return [ln[4:] for ln in rsp.choices[0].message.content.splitlines()
                    if ln.startswith("### ")]
        except Exception as e:
            if i == retry - 1:
                raise
            print(f"retry {i + 1}/{retry} because {e}")
            time.sleep(2 * (i + 1))


# ------------- 主流程 -------------
for fp in SRC_DIR.glob("*.txt"):
    out_path = OUT_DIR / f"{fp.stem}.jsonl"
    if out_path.exists():
        print(f"skip {fp.name} (already segmented_narrativeqa)")
        continue
    done = {json.loads(l)["seg_id"] for l in out_path.open()} if out_path.exists() else set()

    text = fp.read_text(encoding="utf-8", errors="ignore")
    start, seg_id = 0, 0
    with out_path.open("a", encoding="utf-8") as fout:
        while start < len(text):
            if seg_id in done:  # 已完成跳过
                slice_, start = next_slice(text, start)
                seg_id += 1
                continue

            slice_, start = next_slice(text, start)
            chunks = call_llm(slice_)
            json.dump({"seg_id": seg_id, "chunks": chunks}, fout, ensure_ascii=False)
            fout.write("\n");
            fout.flush()
            print(f"{fp.stem} seg {seg_id} -> {len(chunks)} chunks")
            seg_id += 1

print("\n全部文件切分完成！结果已写入 segmented_narrativeqa/*.jsonl")
