import nltk
import numpy as np
import pandas as pd
from nltk.translate.bleu_score import SmoothingFunction
from nltk.translate.bleu_score import corpus_bleu as nltk_corpus_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from sacrebleu import corpus_bleu

for res in ("punkt", "wordnet", "omw-1.4"):
    try:
        nltk.data.find(f"corpora/{res}")
    except LookupError:
        nltk.download(res, quiet=True)

# df = pd.read_csv("../res_narrativeqa/predictions_top5_300.csv")
# df = pd.read_csv("../res_narrativeqa/predictions_top10_300.csv")
# df = pd.read_csv("predictions_dynK.csv")
# df = pd.read_csv("predictions_dynK10.csv")
# df = pd.read_csv("predictions_queryexpand.csv")
# df = pd.read_csv("../res_narrativeqa/predictions_top5_300_dot0.csv")
# df = pd.read_csv("../res_narrativeqa/predictions_top5_300_dot3.csv")
# df = pd.read_csv("../res_narrativeqa/predictions_top10_300_dot0.csv")
# df = pd.read_csv("../res_narrativeqa/predictions_top5_300_win1.csv")
# df = pd.read_csv("../res_narrativeqa/pred_bm25_top5.csv")
# df = pd.read_csv("../res_narrativeqa/pred_sbert_top5.csv")
# df = pd.read_csv("../res_narrativeqa/pred_full_300.csv")
# df = pd.read_csv("../res_narrativeqa/pred_bm25_all.csv")
df = pd.read_csv("../res_narrativeqa/pred_sbert_all.csv")
# df = pd.read_csv("../res_narrativeqa/predictions_top10_300_dot3.csv")
# df = pd.read_csv("../res_narrativeqa/predictions_top10_300_llm.csv")
# df = pd.read_csv("../res_narrativeqa/predictions_queryexpand_new.csv")
preds = df["prediction"].fillna("").astype(str).tolist()
ref1 = df["answer1"].fillna("").astype(str).tolist()
ref2 = df["answer2"].fillna("").astype(str).tolist()
refs_multi = [[r1, r2] for r1, r2 in zip(ref1, ref2)]


def tok(s: str): return s.split()


bleu4 = corpus_bleu(preds, refs_multi).score

smooth = SmoothingFunction().method1
pred_tok = [tok(p) for p in preds]
refs_tok = [[tok(r) for r in refs] for refs in refs_multi]
bleu1 = nltk_corpus_bleu(
    refs_tok, pred_tok,
    weights=(1, 0, 0, 0),
    smoothing_function=smooth
) * 100

# ---------- ROUGE-L (max per candidate) ----------
scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
rouge_vals = [
    max(
        scorer.score(r1, p)["rougeL"].fmeasure,
        scorer.score(r2, p)["rougeL"].fmeasure
    )
    for p, (r1, r2) in zip(preds, refs_multi)
]
rouge_l = np.mean(rouge_vals) * 100

meteor_vals = [
    meteor_score([tok(r1), tok(r2)], tok(p))
    for p, (r1, r2) in zip(preds, refs_multi)
]
meteor = np.mean(meteor_vals) * 100

print(f"BLEU-4 : {bleu4:.2f}")
print(f"BLEU-1 : {bleu1:.2f}")
print(f"ROUGE-L: {rouge_l:.2f}")
print(f"METEOR : {meteor:.2f}")
