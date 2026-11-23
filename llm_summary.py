"""Statistics quoted in entry 02, computed from the saved LLM results.

    python llm_summary.py         # reads results_llm/ and writes two files:

    results_llm_summary.csv
        For each LLM run: AUC, the share of test cases predicted as first-party
        wins, minority-class recall (the share of first-party losses it
        catches), and the observed win rate in the lowest and highest
        predicted-probability fifths. For the stratified runs, the entry 01
        random forest's AUC on the same split, so the two can be compared split
        by split. For the temporal run, a bootstrap interval for the AUC
        (10,000 resamples of the test cases).
    results_llm_stability.csv
        How much the prediction for the same case changes between seeds, for
        the LLM and the random forest, over the cases that fall in two or more
        of the five stratified test folds (full text).

The share of summaries over 512 tokens is not recomputed here; it is in the
runs files in results_llm/. Needs only the base requirements, not a GPU. The
random forest figures are recomputed, so they depend on the scikit-learn
version (see README).
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
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


def quintile_win_rates(y, p):
    order = np.argsort(p)
    fifth = len(order) // 5
    return y[order[:fifth]].mean(), y[order[-fifth:]].mean()


def stability(df, y_all, llm):
    """Range of the predicted probability for the same case across the seeds
    in which it is a test case, and how often the predicted label flips."""
    texts, ids = df["text_full"].to_numpy(), df["ID"].to_numpy()
    rf = []
    for seed in E.SEEDS:
        tr, te = E.split(len(texts), y_all, seed)
        vec = TfidfVectorizer(**TFIDF)
        model = forest(seed).fit(vec.fit_transform(texts[tr]), y_all[tr])
        rf.append(pd.DataFrame({"run": f"seed {seed}", "ID": ids[te],
                                "p": model.predict_proba(vec.transform(texts[te]))[:, 1]}))
    rows = []
    for name, preds in [("Qwen2.5-0.5B + LoRA", llm.rename(columns={"p_first_party_wins": "p"})),
                        ("random forest (entry 01)", pd.concat(rf))]:
        g = preds.groupby("ID")["p"]
        multi = g.count() >= 2
        spread = (g.max() - g.min())[multi]
        flips = g.apply(lambda x: (x >= 0.5).nunique() > 1)[multi]
        rows.append({"model": name, "cases_in_two_or_more_test_folds": int(multi.sum()),
                     "median_range_of_probability": spread.median(),
                     "share_with_range_above_0.5": (spread > 0.5).mean(),
                     "share_where_predicted_label_flips": flips.mean()})
    return pd.DataFrame(rows)


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
            row["bottom_fifth_win_rate"], row["top_fifth_win_rate"] = quintile_win_rates(y, p)
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
    print(f"Mean win rate, lowest fifth {full['bottom_fifth_win_rate'].mean():.2f}, "
          f"highest fifth {full['top_fifth_win_rate'].mean():.2f}")

    stab = stability(df, y_all, pd.read_csv(LLM / "predictions_text_full_stratified.csv"))
    stab.to_csv(ROOT / "results_llm_stability.csv", index=False)
    print("\n" + stab.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
