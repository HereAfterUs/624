# acc_quality.py
import pandas as pd

PRED_CSV = "../res_quality/pred_all.csv"  # 你的预测文件
OUT_CSV = "../res_quality/pred_all_marked.csv"  # 带 is_correct 的输出

# -------- 读取 --------
df = pd.read_csv(PRED_CSV)

# 若列名不同可在这里改
PRED_COL = "prediction"
GOLD_COL = "gold"

# --- 清洗：保留数字预测 ---
good_mask = df[PRED_COL].astype(str).str.fullmatch(r"\d+")
clean_df = df[good_mask].copy()

# 转 int
clean_df[PRED_COL] = clean_df[PRED_COL].astype(int)
clean_df[GOLD_COL] = clean_df[GOLD_COL].astype(int)

# 标记是否正确
clean_df["is_correct"] = (clean_df[PRED_COL] == clean_df[GOLD_COL])

# -------- 统计 --------
total_all = len(df)
total_valid = len(clean_df)
correct = clean_df["is_correct"].sum()

acc_valid = correct / total_valid * 100  # 忽略 ERR 行
acc_all = correct / total_all * 100  # 把 ERR 行算在分母

print(f"Total rows (all)    : {total_all}")
print(f"Rows w/ valid preds : {total_valid}")
print(f"Correct predictions : {correct}")
print(f"Accuracy (valid set): {acc_valid:.2f}%")
print(f"Accuracy (all rows) : {acc_all:.2f}%")

# -------- 保存标记文件 --------
clean_df.to_csv(OUT_CSV, index=False, quoting=pd.io.formats.csvs.csv.QUOTE_ALL)
print(f"✓ Marked file saved to {OUT_CSV}")
