import re, os, html
from pathlib import Path

SRC_TEXT_DIR = Path("../texts_narrativeqa")
SRC_HTML_DIR = Path("../html_narrativeqa")
OUT_DIR = Path("../processed_narrativeqa")
OUT_DIR.mkdir(exist_ok=True)


# ---------- 工具：清洗函数 ----------
def normalize_whitespace(s: str) -> str:
    """把所有换行/制表等压成单空格，然后连成一行"""
    return re.sub(r"\s+", " ", s).strip()


TAG_RE = re.compile(r"<[^>]+>")  # 匹配任何 HTML 标签


def strip_html(raw: str) -> str:
    """删除标签并反转义 HTML 实体"""
    no_tag = TAG_RE.sub(" ", raw)  # 标签替换为空格防止单词粘连
    return normalize_whitespace(html.unescape(no_tag))


# ---------- 处理 texts_narrativeqa/.txt ----------
for fp in SRC_TEXT_DIR.glob("*.txt"):
    out_path = OUT_DIR / fp.name
    cleaned = normalize_whitespace(fp.read_text(encoding="utf-8", errors="ignore"))
    out_path.write_text(cleaned, encoding="utf-8")
    print(f"✓ cleaned text  -> {out_path}")

# ---------- 处理 html_narrativeqa/.txt ----------
for fp in SRC_HTML_DIR.glob("*.txt"):
    out_path = OUT_DIR / fp.name
    cleaned = strip_html(fp.read_text(encoding="utf-8", errors="ignore"))
    out_path.write_text(cleaned, encoding="utf-8")
    print(f"✓ cleaned html_narrativeqa  -> {out_path}")

print(f"\n全部完成！清洗后的文件已写入 {OUT_DIR.absolute()}")
