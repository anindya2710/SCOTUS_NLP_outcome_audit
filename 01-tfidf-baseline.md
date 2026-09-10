# 01 — TF-IDF baseline audit

**2026-09-09**

## Question

Does a TF-IDF and Random Forest classifier predict which party prevails at the
Supreme Court from the Oyez facts summary, and if so, do its feature
importances say anything about what drives the outcome?

## Expectation before running

Two. First, that the classifier would beat the majority-class rate, since that
is what tutorials built on this dataset report. Second, that some of any
performance would turn out to be leakage, because Oyez summaries are written
after the decision and can carry outcome language.

Both wrong, and the second wrong in a way that took some work to see.

## Method

- Stripped the HTML that Oyez ships in the `facts` field. Without this, markup
  tokens enter the vocabulary.
- Fixed a baseline of 0.651 accuracy (always predict "first party wins") and
  0.500 AUC. These do not move between entries.
- Averaged over five stratified 80/20 splits rather than one. Seed spread is
  around one accuracy point, so single-split numbers here are not measurements.
- Fit the vectorizer on the training fold only.
- Built two reduced versions of the text by deleting whole sentences matching
  outcome language, then additionally procedural posture.
- Ran a temporal split, training on terms up to 2011 and testing after.
- Reported minority-class F1 and AUC alongside accuracy.

## Result

| Model | Accuracy | AUC |
|---|---|---|
| Majority-class baseline | 0.651 | 0.500 |
| Random forest, full text | 0.638 | 0.573 |
| Random forest, outcome removed | 0.634 | 0.571 |
| Random forest, posture removed | 0.637 | 0.555 |
| Logistic regression | 0.648 | 0.569 |
| Logistic regression, boilerplate removed | 0.648 | 0.560 |

Nothing beats the constant. The temporal split gives the same picture: 0.660
against a 0.671 majority rate.

## Reading it

The interesting part is the disagreement between the two columns. AUC sits
around 0.57, reliably above chance, so there is signal. Accuracy sits below the
baseline anyway, because with a 65/35 target a model only improves on a
constant by predicting the minority class, and it needs real confidence to do
that profitably. This one rarely has it. Minority-class recall was 0.15.

Threshold tuning does not fix it. Optimising the threshold on the test set,
which is cheating and so an upper bound, reaches 0.657.

Treating the output as a ranking rather than a label does show the signal:
lowest-probability fifth wins 59% of the time, highest fifth 73%, against a
corpus mean of 65%.

## The leakage question

The strip removed 20.5% of the corpus for outcome language and a further 16.1%
for posture, and changed accuracy by less than a point. So there was no leakage
advantage in the first place — not because the text is clean, but because the
model was not successfully exploiting it.

Outcome terms do sit high in the importance ranking (*held*, *reversed*,
*affirmed* all appear in the top 25). None of it converted into skill.

## The part worth keeping

`feature_importances_` returned a clean, plausible-looking ranking from a model
that loses to a constant. Nothing in the output signals that. Plotted with a
title like "top keywords driving outcomes", it would look like a finding.

The check that catches this is one line: compare against always predicting the
majority class. It should run before anything else, and its result should sit
next to every number reported afterwards.

## Was the boilerplate hypothesis right?

Partly, and not in the way expected. IDF does downweight ubiquitous terms —
*court* appears in 90% of documents and gets an IDF of 1.10 against 5.41 for
*abortion*. But *court* still carries the highest mean weight in the matrix,
because it appears often enough within each document that term frequency
outruns the IDF discount.

So boilerplate does dominate the feature space. Removing it changed nothing,
which rules out the hypothesis that it was crowding out substantive vocabulary.
There was no signal underneath to uncover.

## Model problem or data problem?

Run `python src/diagnostics.py`. Three checks.

**The model overfits badly.**

| | Train acc | Test acc | Train AUC | Test AUC |
|---|---|---|---|---|
| Random forest | 1.000 | 0.638 | 1.000 | 0.573 |
| Logistic regression | 0.752 | 0.648 | 0.955 | 0.569 |

The forest memorises the training set perfectly and generalises barely above
chance. With 1,000 TF-IDF features over 2,630 documents, near-unique term
combinations let it separate any labelling it is handed. The features are
expressive. They are not informative.

Regularisation would close that gap by pulling training performance down, not
by pulling test performance up. Logistic regression is already regularised and
lands at the same test AUC.

**More data would not help.**

| Training cases | Test AUC |
|---|---|
| 526 | 0.552 |
| 1,052 | 0.553 |
| 1,578 | 0.565 |
| 2,104 | 0.573 |
| 2,630 | 0.569 |

Fivefold more data buys 0.017 AUC, and the last two points go backwards inside
the noise. The curve is already asymptotic.

**The label is clean.** Disposition alone predicts the target at 0.968.
Affirmed goes to the first party 2% of the time; reversed, 99%. So
`first_party_winner` is close to a deterministic function of whether the Court
reversed. It is not a noisy or incoherent target.

That result is diagnostic only. Disposition cannot be used as a feature —
knowing the Court reversed is already knowing who won.

Taken together: not the classifier, not the sample size, not the label. The
narrative summaries carry very little information about how the Court will
rule, and everything else follows from that.

## Ruled out

- Random forest at 1,000 and 5,000 features, 100 and 300 trees
- Class-balanced random forest and class-balanced logistic regression
- Linear SVC
- Bigrams with minimum document frequency filtering
- `sublinear_tf` to dampen repeated terms
- `max_df` to drop ubiquitous terms
- A 36-term legal boilerplate stopword list
- Decision threshold tuning
- Structured metadata alone (issue area and narrative length): 0.553
- More training data (flat learning curve)
- A cleaner target (the label is already 96.8% learnable)

## Next

Representation is the remaining suspect, so entry 02 tries a frozen transformer
encoder. If embeddings also plateau near 0.58, that localises the ceiling to
the text itself rather than the encoding, which would be a result rather than a
dead end.

The stronger candidate may still be the target. `first_party_winner` is
positional rather than legal, and mixes cases where the petitioner is a
government, a corporation or an individual. Predicting the disposition directly
is entry 03, and it may matter more than any change of features.
