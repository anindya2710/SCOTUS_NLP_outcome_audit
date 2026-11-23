"""Entry 01: does a TF-IDF classifier beat always predicting the majority class?

    python run_audit.py

Writes results.csv, feature_audit.csv and baseline_comparison.png next to this
file. temporal_audit.py has the temporal split for every model and text
version; supporting_checks.py has the numbers quoted in the write-ups that are
not produced here.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.linear_model import LogisticRegression

import data as D
import evaluation as E

RESULTS = Path(__file__).resolve().parent

TFIDF = dict(stop_words="english", max_features=1000)
TFIDF_WIDE = dict(stop_words="english", max_features=5000)

# Terms that appear in nearly every summary regardless of subject matter.
# Included to test whether boilerplate is crowding out substantive vocabulary.
# 37 terms. Note that the list also contains outcome words (affirmed, reversed,
# remanded, vacated, held, ruled, decided).
BOILERPLATE = {
    "court", "courts", "appeals", "district", "circuit", "federal", "state",
    "states", "supreme", "affirmed", "reversed", "remanded", "vacated", "held",
    "ruled", "case", "cases", "act", "filed", "judge", "judges", "justice",
    "law", "laws", "argued", "decision", "ninth", "trial", "appeal",
    "appealed", "petitioner", "respondent", "plaintiff", "defendant",
    "ruling", "opinion", "decided",
}

# The boilerplate run changes three things at once relative to the plain
# logistic regression: the stop list, sublinear_tf and max_df.
# supporting_checks.py changes them one at a time.
TFIDF_BOILERPLATE = dict(
    stop_words=list(ENGLISH_STOP_WORDS) + sorted(BOILERPLATE),
    max_features=5000, sublinear_tf=True, max_df=0.5,
)


def forest(seed):
    return RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1)


def linear(seed):
    return LogisticRegression(max_iter=3000)


def describe(df, y):
    years = D.term_year(df)
    eras = sorted(df.loc[years.isna(), "term"].unique())
    print(f"{len(df)} cases: {years.notna().sum()} with a term year "
          f"({int(years.min())}-{int(years.max())}) and {years.isna().sum()} "
          f"older cases recorded by era ({', '.join(eras)})")
    print(f"first party prevails in {y.mean():.1%} of cases")

    dup = df[df[D.TEXT].duplicated(keep=False)]
    print(f"duplicate narratives: {df[D.TEXT].duplicated().sum()}")
    for _, g in dup.groupby(D.TEXT):
        print(f"  IDs {', '.join(map(str, g['ID']))} ({g['name'].iloc[0]}), "
              f"labels {', '.join(map(str, D.labels(g)))}; kept, as both labels agree")
    print()


def main():
    df = D.add_variants(D.load())
    y = D.labels(df)
    baseline = E.majority_rate(y)
    describe(df, y)

    for col, share in D.retention(df).items():
        print(f"  {D.VARIANTS[col]:<28} {share:5.1%} of characters retained")

    rows = [{"model": "majority-class baseline", "accuracy": baseline,
             "accuracy_sd": 0.0, "auc": 0.5, "auc_sd": 0.0, "f1_minority": 0.0,
             "recall_minority": 0.0}]

    # Does the model beat a constant on any version of the text?
    for col, label in D.VARIANTS.items():
        r = E.evaluate(df[col].to_numpy(), y, forest, TFIDF)
        rows.append({"model": f"random forest, {label}", **r})

    # A simpler model, in case the forest is the problem rather than the text.
    rows.append({"model": "logistic regression, full text",
                 **E.evaluate(df["text_full"].to_numpy(), y, linear, TFIDF_WIDE)})

    # Is legal boilerplate crowding out substantive terms?
    rows.append({"model": "logistic regression, boilerplate removed",
                 **E.evaluate(df["text_full"].to_numpy(), y, linear, TFIDF_BOILERPLATE)})

    results = pd.DataFrame(rows)
    print("\n" + results.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    results.to_csv(RESULTS / "results.csv", index=False)

    models = results.iloc[1:]
    best = models.loc[models["accuracy"].idxmax()]
    if best["accuracy"] <= baseline:
        print(
            f"\nNo configuration exceeds {baseline:.3f}. The classifier has no\n"
            "demonstrated skill under accuracy. Feature importances taken from\n"
            "it describe what it happened to fit and cannot be read as an\n"
            "account of what drives outcomes."
        )
    else:
        print(f"\n{best['model']} exceeds baseline by {best['accuracy'] - baseline:+.3f}.")

    # AUC above 0.5 with accuracy below baseline means signal exists but is too
    # weak to move a decision at the default threshold.
    top = models.loc[models["auc"].idxmax()]
    print(f"\nBest AUC {top['auc']:.3f} ({top['model']}) against 0.500 for a coin flip.")

    ranks = E.rank_quintiles(df["text_full"].to_numpy(), y, linear, TFIDF_WIDE)
    print("Observed win rate by predicted-probability quintile "
          "(logistic regression, full text):")
    print(f"  lowest fifth  {ranks['bottom_quintile']:.2f}")
    print(f"  corpus mean   {ranks['overall']:.2f}")
    print(f"  highest fifth {ranks['top_quintile']:.2f}")

    t = E.temporal_evaluate(df, "text_full", y, forest, TFIDF)
    print(f"\nTemporal split at term {t['cutoff_term']} "
          f"({t['n_train']} train, {t['n_test']} test), random forest, full text:")
    print(f"  majority rate {t['test_majority_rate']:.3f}, "
          f"accuracy {t['accuracy']:.3f} ± {t['accuracy_sd']:.3f}, "
          f"AUC {t['auc']:.3f} ± {t['auc_sd']:.3f}")
    print("  Every model and text version: python temporal_audit.py")

    features = E.feature_table(df["text_full"].to_numpy(), y, forest, linear, TFIDF)
    features.to_csv(RESULTS / "feature_audit.csv", index=False)
    print("\nTop terms by tree importance, one fit on the seed-0 training fold "
          "(from a model that does not beat baseline):")
    print(features.head(15).to_string(index=False,
                                      float_format=lambda v: f"{v:.3f}"))

    plot(results, baseline, len(df))
    print(f"\nWrote results.csv, feature_audit.csv, baseline_comparison.png "
          f"to {RESULTS.name}/")


def plot(results, baseline, n_cases):
    """Dot plot of mean accuracy with ± one standard deviation across seeds.

    Dots rather than bars: the differences are around one point, and a bar
    chart would need either a cut axis (which exaggerates them) or an axis
    from zero (which hides them).
    """
    ink, muted, grid = "#1f1f1f", "#6b6b6b", "#e3e3e3"
    labels = [
        "Random forest, full text",
        "Random forest, outcome language removed",
        "Random forest, outcome + posture removed",
        "Logistic regression, full text",
        "Logistic regression, boilerplate removed",
    ]
    models = results.iloc[1:]
    pos = np.arange(len(models))[::-1]

    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.axvline(baseline, color=muted, lw=1.2, ls="--", zorder=1)
    ax.text(baseline - 0.0006, len(models) - 0.3,
            f"Always predict \"first party wins\": {baseline:.3f}",
            color=muted, fontsize=9, ha="right", va="center")
    ax.errorbar(models["accuracy"], pos, xerr=models["accuracy_sd"], fmt="o",
                color=ink, ecolor=ink, elinewidth=1.2, capsize=0, ms=7, zorder=3)
    for x, p in zip(models["accuracy"], pos):
        ax.text(x, p + 0.22, f"{x:.3f}", color=ink, fontsize=9, ha="center")

    ax.set_yticks(pos, labels, color=ink)
    ax.set_ylim(-0.6, len(models))
    ax.set_xlim(0.615, 0.665)
    ax.set_xlabel("Test accuracy (mean ± sd of five stratified splits)", color=ink)
    ax.xaxis.grid(True, color=grid, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=muted, length=0)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(grid)
    fig.suptitle(
        "No configuration beats a constant predictor\n"
        f"Supreme Court outcome from Oyez case summaries, {n_cases:,} cases",
        fontsize=11, color=ink, x=0.015, ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(RESULTS / "baseline_comparison.png", dpi=150, facecolor="white")


if __name__ == "__main__":
    main()
