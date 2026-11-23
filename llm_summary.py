"""Statistics quoted in entry 02, computed from the saved LLM results.

    python llm_summary.py         # reads results_llm/, writes results_llm_summary.csv

For each LLM run: AUC, the share of test cases predicted as first-party wins,
and minority-class recall (the share of first-party losses it catches). For
the stratified runs, the entry 01 random forest's AUC on the same split, so the
two can be compared split by split. For the temporal run, a bootstrap interval
for the AUC (10,000 resamples of the test cases).

Needs only the base requirements, not a GPU. The random forest figures are
recomputed, so they depend on the scikit-learn version (see README).
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import recall_score, roc_auc_score

import data as D
import evaluation as E
from run_audit import TFIDF, forest

ROOT = Path(__file__).resolve().parent
LLM = ROOT / "results_llm"


def bootstrap_auc(y, p, n=10_000, seed=0):
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(i):
            stats.append(roc_auc_score(y[i], p[i]))
    return np.percentile(stats, [2.5, 97.5])


def main():
    df = D.add_variants(D.load())
    y_all = D.labels(df)
    ids = df["ID"].to_numpy()
    rows = []
    for path in sorted(LLM.glob("predictions_*.csv")):
        tag = path.stem.removeprefix("predictions_")
        variant, split = tag.rsplit("_", 1)
        preds = pd.read_csv(path)
        for run, g in preds.groupby("run", sort=False):
            y, p = g["y"].to_numpy(), g["p_first_party_wins"].to_numpy()
            row = {"variant": variant, "split": split, "run": run, "n_test": len(g),
                   "auc": roc_auc_score(y, p),
                   "share_predicted_first_party_win": (p >= 0.5).mean(),
                   "recall_minority": recall_score(y, (p >= 0.5).astype(int), pos_label=0)}
            if split == "stratified":
                seed = int(run.split()[-1])
                _, te = E.split(len(df), y_all, seed)
                row["same_test_cases_as_entry_01"] = set(ids[te]) == set(g["ID"])
                if variant == "text_full":
                    rf = E.evaluate(df["text_full"].to_numpy(), y_all, forest, TFIDF, seeds=(seed,))
                    row["random_forest_auc_same_split"] = rf["auc"]
            else:
                low, high = bootstrap_auc(y, p)
                row["auc_95_low"], row["auc_95_high"] = low, high
            rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "results_llm_summary.csv", index=False)
    print(out.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    full = out[(out["variant"] == "text_full") & (out["split"] == "stratified")]
    wins = (full["random_forest_auc_same_split"] > full["auc"]).sum()
    print(f"\nRandom forest AUC higher than the LLM's on {wins} of {len(full)} stratified splits.")


if __name__ == "__main__":
    main()
