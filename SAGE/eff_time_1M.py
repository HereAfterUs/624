# eff_time_single.py
import random
import tiktoken
import time
from pathlib import Path

from openai import OpenAI

# ---------- 必填 ----------
API_KEY = "sk-yRuc2168ce331878ed754bd7df810d634b2c3403705vf1a6"  # OpenAI key
BASE_URL = "https://api.gptsapi.net/v1"
# --------------------------

PROCESSED_DIR = Path("processed_narrativeqa")
SLICE_TOK = 10_000
PROMPT_TMPL = """You are a segmentation assistant.
Split into ~100–300 GPT-token chunks.

### chunk 1
### chunk 2
(no markdown)

<<<
{passage}
>>>
"""

enc = tiktoken.get_encoding("cl100k_base")
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# 1) 随机抽 1 篇
files = sorted(PROCESSED_DIR.glob("*.txt"))
pick = random.choice(files)
text = pick.read_text(encoding="utf-8", errors="ignore")
tok_n = len(enc.encode(text))
print(f"Selected file  : {pick.name}")
print(f"Total tokens   : {tok_n:,}")

# 2) 固定 token 切片
ids = enc.encode(text)
slices = [enc.decode(ids[i:i + SLICE_TOK]) for i in range(0, len(ids), SLICE_TOK)]
print(f"Slices to send : {len(slices)}  (each {SLICE_TOK} tok)")

# 3) 计时
t0 = time.perf_counter()
total_chunks = 0
for sl in slices:
    rsp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": PROMPT_TMPL.format(passage=sl)}],
        temperature=0.3,
        timeout=180,
    )
    total_chunks += sum(1 for ln in rsp.choices[0].message.content.splitlines()
                        if ln.startswith("### "))
elapsed = time.perf_counter() - t0

# 4) 报告
print("\n== Efficiency Report ==")
print(f"File            : {pick.name}")
print(f"Tokens          : {tok_n:,}")
print(f"LLM chunks      : {total_chunks}")
print(f"Wall-time (s)   : {elapsed:.1f}")
print(f"Throughput      : {tok_n / elapsed:,.0f} tokens/sec")
