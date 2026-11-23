"""Temporal split for every text version and both models, with accuracy and AUC.

Train on terms up to the 80th-percentile term (2011), test on the terms after
it, exactly as evaluation.temporal_evaluate does. Each row names its model and
its text version, so no number has to be read out of context.

    python temporal_audit.py      # writes results_temporal.csv
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import data as D
import evaluation as E
from run_audit import TFIDF, TFIDF_WIDE, forest, linear

MODELS = {
    "random forest (TF-IDF, 1,000 features)": (forest, TFIDF),
    "logistic regression (TF-IDF, 5,000 features)": (linear, TFIDF_WIDE),
}


def main():
    df = D.add_variants(D.load())
    y = D.labels(df)
    rows = []
    for col, text_label in D.VARIANTS.items():
        for model_label, (factory, vec) in MODELS.items():
            r = E.temporal_evaluate(df, col, y, factory, vec)
            rows.append({"model": model_label, "text": text_label, **r})
    out = pd.DataFrame(rows)
    out.to_csv(Path(__file__).resolve().parent / "results_temporal.csv", index=False)
    print(out.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
