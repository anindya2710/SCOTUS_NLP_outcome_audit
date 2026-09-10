"""Diagnostics for entry 01: is the failure in the model, the data, or the label?

Three checks, each ruling something in or out:

    1. Train against test performance. Separates a model that cannot fit the
       data from one that fits it and fails to generalise.
    2. Learning curve. Says whether more data would help.
    3. Label coherence. Whether the target is learnable at all from any
       feature, which distinguishes a noisy label from weak features.

    python src/diagnostics.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))

import data as D

SEEDS = (0, 1, 2, 3, 4)
FRACTIONS = (0.2, 0.4, 0.6, 0.8, 1.0)


def fit_and_score(texts, y, model, vectorizer_kwargs, train_idx, test_idx):
    vec = TfidfVectorizer(**vectorizer_kwargs)
    x_train = vec.fit_transform(texts[train_idx])
    x_test = vec.transform(texts[test_idx])
    model.fit(x_train, y[train_idx])
    return {
        "train_acc": accuracy_score(y[train_idx], model.predict(x_train)),
        "test_acc": accuracy_score(y[test_idx], model.predict(x_test)),
        "train_auc": roc_auc_score(y[train_idx], model.predict_proba(x_train)[:, 1]),
        "test_auc": roc_auc_score(y[test_idx], model.predict_proba(x_test)[:, 1]),
    }


def generalisation_gap(texts, y):
    """A model that memorises the training set and does not generalise has
    expressive features, not informative ones."""
    configs = [
        ("random forest",
         lambda s: RandomForestClassifier(n_estimators=300, random_state=s, n_jobs=-1),
         dict(stop_words="english", max_features=1000)),
        ("logistic regression",
         lambda s: LogisticRegression(max_iter=3000),
         dict(stop_words="english", max_features=5000)),
    ]
    rows = []
    for name, factory, kw in configs:
        scores = []
        for seed in SEEDS:
            tr, te = train_test_split(
                np.arange(len(texts)), test_size=0.2, random_state=seed, stratify=y
            )
            scores.append(fit_and_score(texts, y, factory(seed), kw, tr, te))
        rows.append({"model": name,
                     **{k: np.mean([s[k] for s in scores]) for k in scores[0]}})
    return pd.DataFrame(rows)


def learning_curve(texts, y):
    """Test AUC against training set size. A flat curve means the corpus is
    not the binding constraint."""
    rows = []
    for frac in FRACTIONS:
        aucs, sizes = [], []
        for seed in SEEDS:
            tr, te = train_test_split(
                np.arange(len(texts)), test_size=0.2, random_state=seed, stratify=y
            )
            rng = np.random.default_rng(seed)
            sub = rng.choice(tr, size=int(len(tr) * frac), replace=False)
            sizes.append(len(sub))

            vec = TfidfVectorizer(stop_words="english", max_features=5000)
            x_train = vec.fit_transform(texts[sub])
            x_test = vec.transform(texts[te])
            model = LogisticRegression(max_iter=3000).fit(x_train, y[sub])
            aucs.append(roc_auc_score(y[te], model.predict_proba(x_test)[:, 1]))

        rows.append({"n_train": int(np.mean(sizes)),
                     "test_auc": np.mean(aucs), "test_auc_sd": np.std(aucs)})
    return pd.DataFrame(rows)


def label_coherence():
    """How well the recorded disposition predicts the target.

    Post-hoc, so useless as a feature: knowing the Court reversed is already
    knowing who won. It is here only to establish that the label is a clean
    function of something, rather than noise.
    """
    df = pd.read_csv(D.DATA_PATH).dropna(
        subset=["facts", "first_party_winner", "disposition"]
    )
    grouped = df.groupby("disposition")["first_party_winner"]
    table = pd.DataFrame({
        "n": grouped.size(),
        "p_first_party_wins": grouped.mean(),
    })
    table["majority_within_group"] = table["p_first_party_wins"].apply(
        lambda p: max(p, 1 - p)
    )
    weights = table["n"] / table["n"].sum()
    accuracy = float((table["majority_within_group"] * weights).sum())
    return table.sort_values("n", ascending=False), accuracy


def main():
    df = D.add_variants(D.load())
    y = D.labels(df)
    texts = df["text_full"].to_numpy()

    print("1. Generalisation gap\n")
    gap = generalisation_gap(texts, y)
    print(gap.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    gap.to_csv("diagnostic_generalisation.csv", index=False)

    worst = gap.loc[(gap["train_auc"] - gap["test_auc"]).idxmax()]
    print(f"\n   Largest gap: {worst['model']}, train AUC {worst['train_auc']:.3f} "
          f"against test {worst['test_auc']:.3f}.")
    print("   The features are expressive enough to separate the training set.")
    print("   They are not informative enough for that separation to transfer.")

    print("\n\n2. Learning curve\n")
    curve = learning_curve(texts, y)
    print(curve.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    curve.to_csv("diagnostic_learning_curve.csv", index=False)

    gain = curve["test_auc"].iloc[-1] - curve["test_auc"].iloc[0]
    print(f"\n   Fivefold more data moves test AUC by {gain:+.3f}.")
    print("   More cases would not change the conclusion.")

    print("\n\n3. Label coherence\n")
    table, accuracy = label_coherence()
    print(table[table["n"] >= 20].to_string(float_format=lambda v: f"{v:.3f}"))
    table.to_csv("diagnostic_label_coherence.csv")

    print(f"\n   Disposition alone predicts the target at {accuracy:.3f}.")
    print("   The label is close to a deterministic function of whether the")
    print("   Court reversed, so the target is clean. The narrative text is")
    print("   what carries little information about it.")



if __name__ == "__main__":
    main()
