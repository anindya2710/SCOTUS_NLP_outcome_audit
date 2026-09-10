"""Evaluation protocol.

Two rules apply to every entry in this project:

    1. Accuracy is always reported next to the majority-class rate. A model
       that cannot beat a constant has not been shown to work.
    2. Nothing is reported from a single split. Seed-to-seed variation on this
       corpus is roughly one accuracy point, so a single number is not a
       measurement.

AUC is carried alongside accuracy because the two can disagree: on an
imbalanced target a model can rank better than chance while still losing to a
constant on accuracy.
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

SEEDS = (0, 1, 2, 3, 4)
TEST_SIZE = 0.2


def majority_rate(y):
    """Accuracy of always predicting the more common label."""
    return max(y.mean(), 1 - y.mean())


def split(n, y, seed):
    return train_test_split(
        np.arange(n), test_size=TEST_SIZE, random_state=seed, stratify=y
    )


def evaluate(texts, y, model_factory, vectorizer_kwargs=None, seeds=SEEDS):
    """Fit and score across seeds. Returns a dict of means and spreads.

    The vectorizer is fit on the training fold only. Fitting on the full
    corpus first leaks test-set vocabulary and document frequencies.
    """
    vectorizer_kwargs = vectorizer_kwargs or {}
    acc, auc, f1 = [], [], []

    for seed in seeds:
        tr, te = split(len(texts), y, seed)
        vec = TfidfVectorizer(**vectorizer_kwargs)
        x_train = vec.fit_transform(texts[tr])
        x_test = vec.transform(texts[te])

        model = model_factory(seed).fit(x_train, y[tr])
        pred = model.predict(x_test)

        acc.append(accuracy_score(y[te], pred))
        f1.append(f1_score(y[te], pred, pos_label=0))  # minority class
        if hasattr(model, "predict_proba"):
            auc.append(roc_auc_score(y[te], model.predict_proba(x_test)[:, 1]))

    return {
        "accuracy": np.mean(acc),
        "accuracy_sd": np.std(acc),
        "auc": np.mean(auc) if auc else np.nan,
        "auc_sd": np.std(auc) if auc else np.nan,
        "f1_minority": np.mean(f1),
    }


def temporal_evaluate(df, text_col, y, model_factory, vectorizer_kwargs=None,
                      quantile=0.8):
    """Train on earlier terms, test on later ones.

    A random split lets the model train on a mixture of eras and test on the
    same mixture. Splitting on time is the harder and more realistic check.
    """
    vectorizer_kwargs = vectorizer_kwargs or {}
    d = df.copy()
    d["term_num"] = pd.to_numeric(d["term"], errors="coerce")
    keep = d["term_num"].notna().to_numpy()
    d, y = d[keep], y[keep]

    cutoff = d["term_num"].quantile(quantile)
    train = (d["term_num"] <= cutoff).to_numpy()
    test = ~train

    vec = TfidfVectorizer(**vectorizer_kwargs)
    x_train = vec.fit_transform(d.loc[train, text_col])
    x_test = vec.transform(d.loc[test, text_col])
    model = model_factory(0).fit(x_train, y[train])

    return {
        "cutoff_term": int(cutoff),
        "n_train": int(train.sum()),
        "n_test": int(test.sum()),
        "test_majority_rate": majority_rate(y[test]),
        "accuracy": accuracy_score(y[test], model.predict(x_test)),
    }


def rank_quintiles(texts, y, model_factory, vectorizer_kwargs=None, seeds=SEEDS):
    """Observed win rate in the lowest and highest predicted-probability fifths.

    Classification asks whether a case will be won. Ranking asks which cases
    are most likely to go a given way. A model can be useless at the first and
    still informative at the second, which is the relevant question if the
    output is meant to order a queue rather than label it.
    """
    vectorizer_kwargs = vectorizer_kwargs or {}
    bottom, top, overall = [], [], []

    for seed in seeds:
        tr, te = split(len(texts), y, seed)
        vec = TfidfVectorizer(**vectorizer_kwargs)
        x_train = vec.fit_transform(texts[tr])
        x_test = vec.transform(texts[te])
        proba = model_factory(seed).fit(x_train, y[tr]).predict_proba(x_test)[:, 1]

        order = np.argsort(proba)
        fifth = len(order) // 5
        bottom.append(y[te][order[:fifth]].mean())
        top.append(y[te][order[-fifth:]].mean())
        overall.append(y[te].mean())

    return {
        "bottom_quintile": np.mean(bottom),
        "top_quintile": np.mean(top),
        "overall": np.mean(overall),
    }


def feature_table(texts, y, model_factory, linear_factory, vectorizer_kwargs=None,
                  top_n=25):
    """Terms the model leans on, with a direction attached.

    Tree importances are unsigned; a linear model fit on the same matrix says
    which way each term pushes. Both are needed to read the list as anything
    other than a ranking.
    """
    vectorizer_kwargs = vectorizer_kwargs or {}
    tr, _ = split(len(texts), y, seeds := SEEDS[0])
    vec = TfidfVectorizer(**vectorizer_kwargs)
    x_train = vec.fit_transform(texts[tr])

    forest = model_factory(seeds).fit(x_train, y[tr])
    linear = linear_factory(seeds).fit(x_train, y[tr])

    return (
        pd.DataFrame(
            {
                "term": vec.get_feature_names_out(),
                "tree_importance": forest.feature_importances_,
                "linear_coefficient": linear.coef_[0],
                "document_frequency": np.asarray((x_train > 0).mean(axis=0)).ravel(),
            }
        )
        .sort_values("tree_importance", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
