# 01 — TF-IDF baseline audit

**2026-09-09** (corrected 2026-11-23; see Corrections at the end)

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
  tokens enter the vocabulary. (Escaped characters such as `&amp;` were missed;
  see Corrections.)
- Fixed a baseline of 0.651 accuracy (always predict "first party wins") and
  0.500 AUC. These do not move between entries.
- Averaged over five stratified 80/20 splits rather than one. Seed spread is
  around one accuracy point, so single-split numbers here are not measurements.
- Fit the vectorizer on the training fold only.
- Random forest: 300 trees on 1,000 TF-IDF features. Logistic regression:
  5,000 features.
- Built two reduced versions of the text by deleting whole sentences matching
  outcome language, then additionally procedural posture.
- Ran a temporal split, training on terms up to 2011 and testing after. The 57
  cases dated only by era (1789–1955) are left out of it.
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

Nothing beats the constant. The temporal split gives the same picture: no
model or text version reaches the 0.671 majority rate of the later terms. On
the full text, the random forest reaches 0.661 (AUC 0.555, mean of five seeds;
`results_temporal.csv`).

## Reading it

The interesting part is the disagreement between the two columns. AUC sits
around 0.57, reliably above chance, so there is signal. Accuracy sits below the
baseline anyway, because with a 65/35 target a model only improves on a
constant by predicting the minority class, and it needs real confidence to do
that profitably. This one rarely has it: the random forest catches 13% of the
cases the first party loses, and logistic regression 5%.

Threshold tuning does not fix it. Optimising the threshold on the test set,
which is cheating and so an upper bound, takes logistic regression to 0.657.

Treating the logistic regression's output as a ranking rather than a label
does show the signal: lowest-probability fifth wins 59% of the time, highest
fifth 73%, against a corpus mean of 65%.

## The leakage question

The strip removed 20.5% of the corpus for outcome language and a further 16.1%
for posture, and changed accuracy by less than a point.

Reading the removed sentences more closely changes what this shows. Of the
4,912 sentences removed as outcome language, 3,582 (73%) name a lower court
and not the Supreme Court, as in "The Court of Appeals affirmed", and 1,062
(22%) name no court at all. Only 268 (5%) mention the Supreme Court. Most of
those cite one of its earlier decisions or describe an earlier step in the
same case, such as granting review or sending the case back; a few describe
the ruling itself. Most of what was removed is therefore procedural history,
which is known before the Supreme Court rules, and is not leakage.

The model's coefficients agree. *Reversed*, *held* and *affirmed* all push the
logistic regression toward a first-party win. For *affirmed*, that only makes
sense as a lower court's affirmance, which the Supreme Court usually reverses;
when the Supreme Court itself affirms, the first party wins 2% of the time.

The check still sets a bound. Any sentence that does give away the Supreme
Court's ruling is among those removed, and removing them all changed nothing.
So whatever post-decision wording the summaries contain, the model was not
exploiting it. What the check cannot show is that the text is free of it.

Outcome terms do sit high in the importance ranking (*held*, *reversed*,
*affirmed* are in the top 25 on all five seeds). None of it converted into
skill.

## The part worth keeping

`feature_importances_` returned a clean, plausible-looking ranking from a model
that loses to a constant. Nothing in the output signals that. Plotted with a
title like "top keywords driving outcomes", it would look like a finding.

The check that catches this is one line: compare against always predicting the
majority class. It should run before anything else, and its result should sit
next to every number reported afterwards.

## Was the boilerplate hypothesis right?

Partly, and not in the way expected. IDF does downweight ubiquitous terms:
*court* appears in 90% of documents and gets an IDF of 1.10, against 5.41 for
*abortion*. But *court* still carries the highest mean weight in the matrix.
The reason is how many documents it appears in, not how heavily it is weighted
where it appears. In the summaries that contain it, *abortion* weighs about
four times as much as *court* does (0.24 against 0.06), but it appears in 1% of
them and *court* in 90%.

So boilerplate does dominate the feature space. Removing it changed little,
which argues against the hypothesis that it was crowding out substantive
vocabulary. One caveat: that run changed three things at once (the 37-term
stop list, `sublinear_tf` and `max_df`), and the list also contains seven
outcome words. Taken one change at a time, none moves accuracy by more than
0.004 or AUC by more than 0.01 (`results_ruled_out.csv`).

## Model problem or data problem?

Run `python diagnostics.py`. Three checks.

**The model overfits badly.**

| | Train acc | Test acc | Train AUC | Test AUC |
|---|---|---|---|---|
| Random forest (1,000 features) | 1.000 | 0.638 | 1.000 | 0.573 |
| Logistic regression (5,000 features) | 0.752 | 0.648 | 0.955 | 0.569 |

The forest memorises the training set perfectly and generalises barely above
chance. With 1,000 TF-IDF features over 2,630 documents, near-unique term
combinations let it separate any labelling it is handed. The features are
expressive. They are not informative.

Regularisation would close that gap by pulling training performance down, not
by pulling test performance up. Logistic regression is already regularised and
lands at the same test AUC.

**The learning curve rises slowly.** Logistic regression, each smaller training
set a stratified subsample of the full training fold:

| Training cases | Test AUC | ±sd |
|---|---|---|
| 526 | 0.540 | 0.028 |
| 1,052 | 0.560 | 0.025 |
| 1,578 | 0.557 | 0.021 |
| 2,104 | 0.570 | 0.024 |
| 2,630 | 0.569 | 0.013 |

Five times more data adds 0.03 AUC. The spread across seeds is as large as
most of the steps, so the curve cannot say whether it has levelled off.
Nothing in it suggests a large gain from more cases of the same kind, but
2,630 documents is a small sample for 5,000 features, and a modest gain is not
ruled out.

**The label is clean.** Disposition alone predicts the target at 0.968.
Affirmed goes to the first party 2% of the time; reversed, 99%. So
`first_party_winner` is close to a deterministic function of whether the Court
reversed. It is not a noisy or incoherent target.

That result is diagnostic only. Disposition cannot be used as a feature —
knowing the Court reversed is already knowing who won.

Taken together: not the classifier, and not the label. More data of the same
kind may help a little. The narrative summaries carry very little information
about how the Court will rule, and everything else follows from that.

## Ruled out

All on the full text, mean of five stratified splits; every number is in
`results_ruled_out.csv`.

- Random forest at 1,000 and 5,000 features, 100 and 300 trees: accuracy
  0.633–0.642
- Class-balanced random forest (0.644) and class-balanced logistic regression
  (0.579; it trades accuracy for minority recall, which rises to 0.43)
- Linear SVC: 0.597
- Unigrams and bigrams with a minimum document frequency of 5: 0.647
- `sublinear_tf` alone: 0.652. This is the only configuration above the 0.651
  baseline, by less than one case per split and well inside its spread (±0.006)
- `max_df` = 0.5 alone: 0.648
- The 37-term legal boilerplate stopword list alone: 0.646
- Decision threshold tuning (upper bound 0.657, above)
- Structured metadata alone (issue area and narrative length): 0.646 with
  logistic regression, AUC 0.529
- More training data (slowly rising learning curve, above)
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

*2026-11-23: the disposition plan has been dropped. Disposition and the winner
agree in 96.8% of cases (above), so it would be nearly the same target. The
README has the current plan.*

## Corrections

**2026-11-23.** An outside review of the repository found the following. None
of it changes the conclusion.

- **Temporal split.** The temporal accuracy first reported here (0.660) came
  from the random forest on the outcome + posture removed text, not the full
  text, and the text did not say so. It reproduces with scikit-learn 1.4 or
  earlier; from 1.5 on, the same run gives 0.665. `temporal_audit.py` now runs
  every model and text version, with the random forest averaged over five
  seeds. On the full text, the forest reaches 0.661 (AUC 0.555).
- **Leakage framing.** Most outcome-language sentences describe lower courts,
  so the strip tests less than this entry first claimed (see The leakage
  question).
- **Unlabelled figures.** Several figures did not say which model produced
  them. Each is now labelled.
- **Numbers with no script.** The threshold bound, the recall figure, the IDF
  figures and the Ruled out list now come from `supporting_checks.py`. Two did
  not reproduce. Minority recall is 0.13 for the random forest, not 0.15. The
  structured-metadata result is now accuracy 0.646 and AUC 0.529; the earlier
  0.553 could not be traced to a model.
- **Boilerplate test.** It changed three settings at once, and its stop list
  included outcome words. The one-at-a-time version is in
  `results_ruled_out.csv`.
- **Learning curve.** The smaller training sets were drawn at random rather
  than stratified. They are now stratified, which changed the table, and the
  first reading of it ("already asymptotic", "more data would not help") was
  stronger than the numbers allow.
- **IDF explanation.** Corrected: *court*'s high mean weight comes mainly from
  how many documents it appears in.
- **Text cleaning.** HTML entities such as `&amp;` were not decoded. Several
  word patterns matched more than intended ("affirmative action", "vacation",
  "held in jail"). Sentences were split at abbreviations such as "v." and
  "U.S.". Rerunning with all three fixed (`strict=True` in `data.py`) gives the
  same picture: best accuracy 0.648 against 0.651, AUC 0.547–0.576, and no
  temporal result above 0.671 for any model (`results_cleaning_check.csv`).
  The numbers in this entry are left as originally run.
- **Other.** The data covers 1789–2020, not only 1955–2020: 57 older cases are
  dated by era and are left out of the temporal split. One narrative appears
  twice. The boilerplate list has 37 terms, not 36. All outputs were
  regenerated with scikit-learn 1.4.2, which moved a few spreads in the third
  decimal.
- **Second pass, same day.** A further review found two errors in the code
  added for these corrections. The strict cleaning still split sentences after
  "U.S.C.", because the pattern for "U.S." matched first. The count of
  sentences naming the Supreme Court only caught its full name, while Oyez
  usually writes "the Supreme Court", so the first version of the leakage
  paragraph above said that only five sentences named it. Both are fixed, and
  the figures above are from the corrected code.
