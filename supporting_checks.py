"""Numbers quoted in entry 01 that run_audit.py and diagnostics.py do not produce.

    python supporting_checks.py

Writes three files next to this one:

results_ruled_out.csv
    Every configuration on entry 01's "Ruled out" list, on the full text,
    mean and spread over the five stratified splits. Includes the boilerplate
    test taken apart, one change at a time.
results_cleaning_check.csv
    The entry 01 models rerun on the strict text cleaning (data.load and
    data.add_variants with strict=True), stratified and temporal.
supporting_values.csv
    Single values: the threshold-tuning upper bound, minority-class recall,
    the IDF figures, what the outcome-language strip actually removes, and how
    stable the top of the feature list is across seeds.
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC

import data as D
import evaluation as E
from run_audit import BOILERPLATE, TFIDF, TFIDF_BOILERPLATE, TFIDF_WIDE, forest, linear

RESULTS = Path(__file__).resolve().parent
OUTCOME_WORDS = {"affirmed", "reversed", "remanded", "vacated", "held", "ruled", "decided"}

# Sentences that name a court below the Supreme Court of the United States.
STATES = (
    "alabama|alaska|arizona|arkansas|california|colorado|connecticut|delaware|"
    "florida|georgia|hawaii|idaho|illinois|indiana|iowa|kansas|kentucky|"
    "louisiana|maine|maryland|massachusetts|michigan|minnesota|mississippi|"
    "missouri|montana|nebraska|nevada|new hampshire|new jersey|new mexico|"
    "new york|north carolina|north dakota|ohio|oklahoma|oregon|pennsylvania|"
    "rhode island|south carolina|south dakota|tennessee|texas|utah|vermont|"
    "virginia|washington|west virginia|wisconsin|wyoming|puerto rico|guam|"
    "virgin islands|state"
)
LOWER_COURT = (
    r"court of appeals|circuit|district court|trial court|lower court|"
    r"appellate court|superior court|state court|court of criminal appeals|"
    rf"\b({STATES})('s)? supreme court|supreme court of ({STATES})\b"
)
# A state's highest court. West Virginia's is the "Supreme Court of Appeals".
STATE_SUPREME_COURT = (
    rf"\b({STATES})('s)? supreme court|supreme court of ({STATES})\b|"
    r"supreme court of appeals"
)


def names_us_supreme_court(sentence):
    """Any mention of the Supreme Court that is not a state's supreme court.
    Oyez usually writes plain "the Supreme Court" for the U.S. Supreme Court."""
    rest = re.sub(STATE_SUPREME_COURT, " ", sentence, flags=re.I)
    return re.search(r"supreme court", rest, re.I) is not None


def rf(n_trees, balanced=False):
    return lambda s: RandomForestClassifier(
        n_estimators=n_trees, random_state=s, n_jobs=-1,
        class_weight="balanced" if balanced else None)


def lr(balanced=False):
    return lambda s: LogisticRegression(max_iter=3000,
                                        class_weight="balanced" if balanced else None)


def ruled_out(df, y):
    texts = df["text_full"].to_numpy()
    stops = list(ENGLISH_STOP_WORDS)
    configs = [
        ("random forest, 1,000 features, 100 trees", rf(100), TFIDF),
        ("random forest, 5,000 features, 100 trees", rf(100), TFIDF_WIDE),
        ("random forest, 5,000 features, 300 trees", rf(300), TFIDF_WIDE),
        ("random forest, class-balanced", rf(300, True), TFIDF),
        ("logistic regression, class-balanced", lr(True), TFIDF_WIDE),
        ("linear SVC", lambda s: LinearSVC(random_state=s), TFIDF_WIDE),
        ("logistic regression, unigrams and bigrams, min_df=5", lr(),
         dict(TFIDF_WIDE, ngram_range=(1, 2), min_df=5)),
        ("logistic regression, sublinear_tf only", lr(), dict(TFIDF_WIDE, sublinear_tf=True)),
        ("logistic regression, max_df=0.5 only", lr(), dict(TFIDF_WIDE, max_df=0.5)),
        ("logistic regression, 37-term boilerplate list only", lr(),
         dict(stop_words=stops + sorted(BOILERPLATE), max_features=5000)),
        ("logistic regression, boilerplate list without its 7 outcome words", lr(),
         dict(stop_words=stops + sorted(BOILERPLATE - OUTCOME_WORDS), max_features=5000)),
        ("logistic regression, all three boilerplate changes (as in results.csv)", lr(),
         TFIDF_BOILERPLATE),
    ]
    rows = [{"configuration": name, **E.evaluate(texts, y, factory, kw)}
            for name, factory, kw in configs]
    rows += metadata_only(df, y)
    return pd.DataFrame(rows)


def metadata_only(df, y):
    """Issue area (one-hot, missing as its own category) and log narrative length."""
    d = df.assign(issue=df["issue_area"].fillna("missing"),
                  log_length=np.log(df["text_full"].str.len()))
    rows = []
    for name, factory in [("structured metadata only, logistic regression", lr()),
                          ("structured metadata only, random forest", rf(300))]:
        acc, auc = [], []
        for seed in E.SEEDS:
            tr, te = E.split(len(d), y, seed)
            features = ColumnTransformer([("issue", OneHotEncoder(handle_unknown="ignore"), ["issue"]),
                                          ("length", StandardScaler(), ["log_length"])])
            model = make_pipeline(features, factory(seed)).fit(d.iloc[tr], y[tr])
            acc.append(accuracy_score(y[te], model.predict(d.iloc[te])))
            auc.append(roc_auc_score(y[te], model.predict_proba(d.iloc[te])[:, 1]))
        rows.append({"configuration": name, "accuracy": np.mean(acc), "accuracy_sd": np.std(acc),
                     "auc": np.mean(auc), "auc_sd": np.std(auc)})
    return rows


def cleaning_check():
    df = D.add_variants(D.load(strict=True), strict=True)
    y = D.labels(df)
    rows = []
    for col, label in D.VARIANTS.items():
        rows.append({"split": "stratified", "model": f"random forest, {label}",
                     **E.evaluate(df[col].to_numpy(), y, forest, TFIDF)})
    rows.append({"split": "stratified", "model": "logistic regression, full text",
                 **E.evaluate(df["text_full"].to_numpy(), y, linear, TFIDF_WIDE)})
    rows.append({"split": "stratified", "model": "logistic regression, boilerplate removed",
                 **E.evaluate(df["text_full"].to_numpy(), y, linear, TFIDF_BOILERPLATE)})
    for col, label in D.VARIANTS.items():
        rows.append({"split": "temporal", "model": f"random forest, {label}",
                     **E.temporal_evaluate(df, col, y, forest, TFIDF)})
        rows.append({"split": "temporal", "model": f"logistic regression, {label}",
                     **E.temporal_evaluate(df, col, y, linear, TFIDF_WIDE)})
    rows.append({"split": "temporal", "model": "logistic regression, boilerplate removed",
                 **E.temporal_evaluate(df, "text_full", y, linear, TFIDF_BOILERPLATE)})
    out = pd.DataFrame(rows)
    out.insert(2, "majority_rate", np.where(out["split"] == "stratified", E.majority_rate(y),
                                            out.get("test_majority_rate")))
    extra = {"strict: characters retained, " + D.VARIANTS[c]: share
             for c, share in D.retention(df).items()}
    extra |= {"strict: empty summaries, " + D.VARIANTS[c]: int((df[c] == "").sum())
              for c in D.VARIANTS}
    return out.drop(columns=["test_majority_rate", "cutoff_term"], errors="ignore"), extra


def threshold_upper_bound(texts, y):
    """Best accuracy over thresholds 0.00-1.00 in 0.01 steps, chosen on each
    TEST fold (so an upper bound, not a usable result), averaged over splits."""
    grid = np.round(np.arange(0, 1.001, 0.01), 2)
    out = {}
    for name, factory, kw in [("logistic regression", linear, TFIDF_WIDE),
                              ("random forest", forest, TFIDF)]:
        best = []
        for seed in E.SEEDS:
            tr, te = E.split(len(texts), y, seed)
            vec = TfidfVectorizer(**kw)
            model = factory(seed).fit(vec.fit_transform(texts[tr]), y[tr])
            p = model.predict_proba(vec.transform(texts[te]))[:, 1]
            best.append(max(accuracy_score(y[te], (p >= t).astype(int)) for t in grid))
        out[f"threshold tuned on test fold (upper bound), {name}, full text"] = np.mean(best)
    return out


def idf_figures(texts):
    """IDF and mean TF-IDF weight, vectorizer fit on the whole corpus."""
    vec = TfidfVectorizer(stop_words="english")
    m = vec.fit_transform(texts)
    vocab = vec.vocabulary_
    mean_weight = np.asarray(m.mean(axis=0)).ravel()
    present = np.asarray((m > 0).mean(axis=0)).ravel()
    terms = vec.get_feature_names_out()
    top = terms[mean_weight.argmax()]
    out = {}
    for w in ["court", "abortion"]:
        out[f"IDF, {w}"] = vec.idf_[vocab[w]]
        out[f"share of summaries containing, {w}"] = present[vocab[w]]
        out[f"mean TF-IDF weight among summaries that contain it, {w}"] = (
            m[:, vocab[w]].sum() / (m[:, vocab[w]] > 0).sum())
    out["term with the highest mean TF-IDF weight"] = top
    return out


def outcome_strip(df):
    """What the outcome-language strip removes, in the default cleaning.

    The Supreme Court mentions include earlier rulings cited as precedent and
    earlier steps in the same case (granting review, sending it back), as well
    as some that describe the decision itself; this count does not separate
    them.
    """
    sents = [s for t in df["text_full"] for s in D.split_sentences(t)
             if re.search(D.OUTCOME_TERMS, s, re.I)]
    lower = np.array([bool(re.search(LOWER_COURT, s, re.I)) for s in sents])
    us = np.array([names_us_supreme_court(s) for s in sents])
    n = len(sents)
    return {
        "outcome-language sentences removed": n,
        "of which name a lower court and not the U.S. Supreme Court": int((lower & ~us).sum()),
        "share naming a lower court and not the U.S. Supreme Court": (lower & ~us).mean(),
        "of which mention the U.S. Supreme Court (any wording)": int(us.sum()),
        "share mentioning the U.S. Supreme Court": us.mean(),
        "of which mention the U.S. Supreme Court and also a lower court": int((us & lower).sum()),
        "of which name no court": int((~us & ~lower).sum()),
        "share naming no court": (~us & ~lower).mean(),
    }


def feature_stability(texts, y, top_n=25):
    """How much of the top of the tree-importance list recurs across seeds."""
    tops = []
    for seed in E.SEEDS:
        tr, _ = E.split(len(texts), y, seed)
        vec = TfidfVectorizer(**TFIDF)
        model = forest(seed).fit(vec.fit_transform(texts[tr]), y[tr])
        order = np.argsort(model.feature_importances_)[::-1][:top_n]
        tops.append(set(vec.get_feature_names_out()[order]))
    out = {f"top-{top_n} terms (seed 0) that are in the top {top_n} for all five seeds":
           len(set.intersection(*tops))}
    for w in ["court", "held", "reversed", "affirmed"]:
        out[f"seeds with '{w}' in the top {top_n}"] = sum(w in t for t in tops)
    return out


def main():
    df = D.add_variants(D.load())
    y = D.labels(df)
    texts = df["text_full"].to_numpy()

    values = {}
    for name, factory, kw in [("random forest", forest, TFIDF),
                              ("logistic regression", linear, TFIDF_WIDE)]:
        r = E.evaluate(texts, y, factory, kw)
        values[f"minority-class recall, {name}, full text"] = r["recall_minority"]
    values |= threshold_upper_bound(texts, y)
    values |= idf_figures(texts)
    values |= outcome_strip(df)
    values |= feature_stability(texts, y)

    print("Ruled-out configurations (full text, five stratified splits)...")
    ruled = ruled_out(df, y)
    ruled.to_csv(RESULTS / "results_ruled_out.csv", index=False)
    print(ruled.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    print("\nStrict cleaning...")
    clean, extra = cleaning_check()
    values |= extra
    clean.to_csv(RESULTS / "results_cleaning_check.csv", index=False)
    print(clean.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    out = pd.DataFrame({"check": list(values), "value": list(values.values())})
    out.to_csv(RESULTS / "supporting_values.csv", index=False)
    print("\n" + out.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
