"""Temporal split for every entry 01 model and text version, with accuracy and AUC.

Train on terms up to the 80th-percentile term (2011), test on the terms after
it, exactly as evaluation.temporal_evaluate does. The 57 cases whose term is
recorded as an era rather than a year are left out. Each model is refit with
the five seeds; the spread comes from that alone. Each row names its model and
its text version, so no number has to be read out of context.

    python temporal_audit.py      # writes results_temporal.csv
"""
from pathlib import Path

import pandas as pd

import data as D
import evaluation as E
from run_audit import TFIDF, TFIDF_BOILERPLATE, TFIDF_WIDE, forest, linear

MODELS = {
    "random forest (TF-IDF, 1,000 features)": (forest, TFIDF),
    "logistic regression (TF-IDF, 5,000 features)": (linear, TFIDF_WIDE),
}
# The boilerplate list was only ever applied to the full text.
BOILERPLATE_MODEL = ("logistic regression, boilerplate removed", (linear, TFIDF_BOILERPLATE))


def main():
    df = D.add_variants(D.load())
    y = D.labels(df)
    runs = [(m, col) for col in D.VARIANTS for m in MODELS]
    rows = []
    for model_label, col in runs + [(BOILERPLATE_MODEL[0], "text_full")]:
        factory, vec = MODELS.get(model_label) or BOILERPLATE_MODEL[1]
        r = E.temporal_evaluate(df, col, y, factory, vec)
        rows.append({"model": model_label, "text": D.VARIANTS[col], **r})
    out = pd.DataFrame(rows)
    out.to_csv(Path(__file__).resolve().parent / "results_temporal.csv", index=False)
    print(out.drop(columns=["cutoff_term"]).to_string(
        index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
