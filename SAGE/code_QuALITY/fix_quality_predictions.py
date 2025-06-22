# fix_quality_predictions.py
import _csv

import pandas as pd

IN_CSV = "../res_quality/pred_all_new.csv"
# OUT_CSV = "../res_quality/pred_quality_naive_fix.csv"  # 修正后另存

# ---------- 1. 读取 ----------
df = pd.read_csv(IN_CSV)

# ---------- 2. prediction +1 ----------
df["prediction"] = df["prediction"] + 1  # 0→1, 1→2, 2→3, 3→4
# 若之前用  -1  作为解析失败，可保留不变：
# df.loc[df["prediction"] == 0, "prediction"] = -1

# ---------- 3. 计算 Accuracy ----------
acc = (df["prediction"] == df["gold"]).mean() * 100
print(f"Accuracy  =  {acc:.2f}%")

# ---------- 4. 保存 ----------
# df.to_csv(OUT_CSV, index=False, quoting=_csv.QUOTE_ALL)
# print(f"✓  corrected file written to  {OUT_CSV}")
