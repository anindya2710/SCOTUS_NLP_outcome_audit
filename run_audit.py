"""
    python src/run_audit.py
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent))

import data as D
import evaluation as E

RESULTS = Path(__file__).resolve().parent 

TFIDF = dict(stop_words="english", max_features=1000)
TFIDF_WIDE = dict(stop_words="english", max_features=5000)

# Terms that appear in nearly every summary regardless of subject matter.
# Included to test whether boilerplate is crowding out substantive vocabulary.
BOILERPLATE = {
    "court", "courts", "appeals", "district", "circuit", "federal", "state",
    "states", "supreme", "affirmed", "reversed", "remanded", "vacated", "held",
    "ruled", "case", "cases", "act", "filed", "judge", "judges", "justice",
    "law", "laws", "argued", "decision", "ninth", "trial", "appeal",
    "appealed", "petitioner", "respondent", "plaintiff", "defendant",
    "ruling", "opinion", "decided",
}


def forest(seed):
    return RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1)


def linear(seed):
    return LogisticRegression(max_iter=3000)


def main():
    df = D.add_variants(D.load())
    y = D.labels(df)
    baseline = E.majority_rate(y)

    terms = pd.to_numeric(df["term"], errors="coerce").dropna()
    print(f"{len(df)} cases, terms {int(terms.min())}-{int(terms.max())}")
    print(f"first party prevails in {y.mean():.1%} of cases")
    print(f"duplicate narratives: {df[D.TEXT].duplicated().sum()}\n")

    for col, share in D.retention(df).items():
        print(f"  {D.VARIANTS[col]:<28} {share:5.1%} of characters retained")

    rows = [{"model": "majority-class baseline", "accuracy": baseline,
             "accuracy_sd": 0.0, "auc": 0.5, "auc_sd": 0.0, "f1_minority": 0.0}]

    # Does the model beat a constant on any version of the text?
    for col, label in D.VARIANTS.items():
        r = E.evaluate(df[col].to_numpy(), y, forest, TFIDF)
        rows.append({"model": f"random forest, {label}", **r})

    # A simpler model, in case the forest is the problem rather than the text.
    rows.append({"model": "logistic regression, full text",
                 **E.evaluate(df["text_full"].to_numpy(), y, linear, TFIDF_WIDE)})

    # Is legal boilerplate crowding out substantive terms?
    stops = list(ENGLISH_STOP_WORDS) + sorted(BOILERPLATE)
    rows.append({
        "model": "logistic regression, boilerplate removed",
        **E.evaluate(df["text_full"].to_numpy(), y, linear,
                     dict(stop_words=stops, max_features=5000, sublinear_tf=True,
                          max_df=0.5)),
    })

    results = pd.DataFrame(rows)
    print("\n" + results.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    results.to_csv(RESULTS / "results.csv", index=False)

    best = results.loc[1:, "accuracy"].max()
    if best <= baseline:
        print(
            f"\nNo configuration exceeds {baseline:.3f}. The classifier has no\n"
            "demonstrated skill under accuracy. Feature importances taken from\n"
            "it describe what it happened to fit and cannot be read as an\n"
            "account of what drives outcomes."
        )
    else:
        print(f"\nBest configuration exceeds baseline by {best - baseline:+.3f}.")

    # AUC above 0.5 with accuracy below baseline means signal exists but is too
    # weak to move a decision at the default threshold.
    auc = results.loc[results["model"].str.startswith("logistic"), "auc"].max()
    print(f"\nBest AUC {auc:.3f} against 0.500 for a coin flip.")

    ranks = E.rank_quintiles(df["text_full"].to_numpy(), y, linear, TFIDF_WIDE)
    print("Observed win rate by predicted-probability quintile:")
    print(f"  lowest fifth  {ranks['bottom_quintile']:.2f}")
    print(f"  corpus mean   {ranks['overall']:.2f}")
    print(f"  highest fifth {ranks['top_quintile']:.2f}")

    t = E.temporal_evaluate(df, "text_no_posture", y, forest, TFIDF)
    print(f"\nTemporal split at term {t['cutoff_term']} "
          f"({t['n_train']} train, {t['n_test']} test)")
    print(f"  majority rate {t['test_majority_rate']:.3f}, "
          f"model {t['accuracy']:.3f}")

    features = E.feature_table(df["text_full"].to_numpy(), y, forest, linear, TFIDF)
    features.to_csv(RESULTS / "feature_audit.csv", index=False)
    print("\nTop terms by tree importance "
          "(from a model that does not beat baseline):")
    print(features.head(15).to_string(index=False,
                                      float_format=lambda v: f"{v:.3f}"))

    plot(results, baseline)
    print(f"\nWrote results.csv, feature_audit.csv, baseline_comparison.png "
          f"to {RESULTS.name}/")


def plot(results, baseline):
    labels = [
        "majority\nbaseline", "RF\nfull text", "RF\nno outcome",
        "RF\nno posture", "logistic\nfull text", "logistic\nno boilerplate",
    ]
    values = results["accuracy"].to_numpy()
    errors = results["accuracy_sd"].to_numpy()

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(labels, values, yerr=errors, capsize=4,
           color=["#3d3d3d"] + ["#8c2f2f"] * (len(labels) - 1))
    ax.axhline(baseline, ls="--", lw=1, color="#3d3d3d")
    ax.set_ylim(0.55, 0.70)
    ax.set_ylabel("test accuracy, mean of 5 seeds")
    ax.set_title(
        "No configuration outperforms a constant predictor\n"
        "Supreme Court outcome from Oyez case narratives, n=3,288",
        fontsize=11,
    )
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(RESULTS / "baseline_comparison.png", dpi=150)


if __name__ == "__main__":
    main()
