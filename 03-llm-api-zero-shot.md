# 03 — Zero-shot commercial LLM through its API

**2026-11-24** (question, expectation and settings fixed before the full runs)

**2026-11-24** (results added)

**2026-11-24** (corrected; see Corrections at the end)

## Question

Can a commercial language model, given a case summary and no training on this
data, predict which party wins more often than the majority rate? And how much
of what it gets right comes from remembering the case rather than reading the
summary?

Entry 02 fine-tuned a 0.5-billion-parameter model and found no gain over
TF-IDF. Before moving to a frozen encoder, I ask a commercial model directly,
with no training at all. This tests the summaries with a different kind of
reader at a cost of a few cents. It is also where memorization is most likely,
as such models are trained on public text, and every case here was decided
years before this model's training cutoff (June 2026). Entry 02 planned a
memorization check for reading any gain; this entry builds that check in from
the start. The frozen encoder is now entry 04, and the later plans move down
by one. Entries 01 and 02 are left as they were written.

## Expectation before running

From the case name alone, the model will answer most of the 638 cases and beat
the majority rate on them, as the outcomes of Supreme Court cases from 2012
onward are widely reported.

Given the summary, its accuracy will also be above the majority rate of 0.671.
However, I expect most of that gain to come from memory: on the cases where the
name-only answer is wrong or unknown, accuracy from the summary should fall to
about the majority rate of those cases. If it stays clearly above it there,
that would be the first sign in this log that the summaries themselves carry
usable signal.

A gain from the summaries is not read as skill unless it holds on the cases the
model does not recognize by name.

## Method

Fixed before the full runs (`llm_api_baseline.py`):

- Model `claude-haiku-5-5` through the Anthropic Messages API (Python SDK
  `anthropic`), with thinking disabled, effort `low` and at most 10 output
  tokens. The model does not accept a temperature setting, so its answers can
  differ between runs. Each prompt is run once, and every answer is saved with
  the model ID and the date, as a hosted model can be changed or retired (this
  one not before October 2027).
- Test set: the 638 cases of the temporal split (terms after 2011). The model
  is shown no labels and no training cases, so there is nothing to tune.
- Two prompts, with the exact wording in `PROMPTS`:
  - `facts`: the summary and the two parties' names. The names are needed so
    that the model can say which party won; they also let it recognize the case.
  - `name`: the case name, term and parties only, with "unknown" allowed as an
    answer.
- Text: the strict cleaning in `data.py`. Entries 01 and 02 used the default
  cleaning; for the full text, the difference is decoded HTML entities such as
  `&amp;`.
- Measures: accuracy against the majority rate of the same cases (0.671 for
  all 638); the share of cases given to the first party; and accuracy from the
  summaries split by the name-only answer for the same case (correct, wrong or
  unknown), with the majority rate of each group. There is no AUC, as the model
  returns a label rather than a probability.

Before the full runs, a 20-case run checked that the API calls worked. It
scored 0.700 against a majority rate of 0.600 on those 20 cases, which is too
few to mean anything. Nothing that affects the answers was changed after it;
the memory split was added to the reporting. Its 20 answers are kept and form
part of the full run.

Cost: the 20 cases used 11,677 input tokens, about 580 per case, so the full
summary run comes to about 370,000 input tokens, roughly $0.04 at the listed
$0.10 per million. The name-only run is smaller.

## Result

One run of each prompt on the 638 later cases. Every answer is in
`results_llm_api/facts.jsonl` and `results_llm_api/name.jsonl`, and the figures
below are in `summary.csv`, `comparison.csv` and `memory_split.csv` in the same
folder.

| Prompt | Cases answered | Accuracy | Majority rate (same cases) | Share given to the first party |
|---|---|---|---|---|
| Summary and parties (`facts`) | 638 of 638 | 0.759 | 0.671 | 0.652 |
| Case name, term and parties only (`name`) | 597 of 638 | 0.747 | 0.678 | 0.784 |

From the name alone, the model answered "unknown" for 39 cases, and 2 of its
replies could not be read: one was cut off at the 10-token limit ("Law v.
Siegel was dec") and the other was "Split". Both are counted as unknown below.

On all 638 cases, with any reply other than "first" or "second" counted as
"first", the more common winner:

| Predictor | Correct | Accuracy |
|---|---|---|
| Always the first party | 428 | 0.671 |
| Name only | 469 | 0.735 |
| Summary | 484 | 0.759 |

Compared in pairs on the same cases, counting the cases that only one of the
two gets right (exact McNemar test):

| Comparison | Only the first right | Only the second right | p |
|---|---|---|---|
| Summary against always the first party | 139 | 83 | 0.0002 |
| Name only against always the first party | 85 | 44 | 0.0004 |
| Summary against name only | 94 | 79 | 0.29 |

Accuracy from the summaries, split by the name-only answer for the same case,
next to the first party's win rate in each group (the accuracy of always
answering "first" there):

| Name-only answer | Cases | Accuracy from the summary | First party's win rate |
|---|---|---|---|
| Correct | 446 | 0.843 | 0.809 |
| Wrong | 151 | 0.576 | 0.291 |
| Unknown | 41 | 0.512 | 0.561 |

Both runs together used 485,197 input tokens and 4,292 output tokens, about
$0.05 at the listed prices.

## Reading it

This is the first model in this log to clearly beat the majority rate. From the
summaries, it is right on 484 of the 638 later cases (0.759), against 428
(0.671) for always answering "first party", and a paired test puts that well
beyond chance (p = 0.0002). However, most of that gain does not need the
summary. From the case name, term and parties alone, it is right on 469 cases
(0.735), so 41 of the 56 extra correct cases, or 6.4 of the 8.8 points, are
already there.

The summary adds 15 more correct cases (2.4 points). Where the two prompts
disagree, the summary is right on 94 cases and the name alone on 79, a
difference that a paired test cannot tell from chance (p = 0.29). So this run
cannot say whether the summaries add anything at all. If they do, the gain is
small, and it may come from reading the summary or from details in it that help
the model recall the case.

The split by the name-only answer says less than it seems to. Two of its groups
are defined partly by the outcome: where the name-only answer was wrong, any
different answer from the summary is a correct one, and where it was right, any
different answer is a wrong one. The summary's 87 correct answers in the
"wrong" group (0.576, against 0.291 for always answering "first") have to be
read together with the 70 cases it gets wrong in the "correct" group, and the
paired comparison above is the fair test of the two prompts. Only the 41 "unknown"
cases form a group that does not depend on the outcome. There, the summary is
right on 21 against 23 for always answering "first" (0.512 against 0.561), no
better than the constant, though 41 cases are too few to be sure.

Against the expectation: the model did beat the majority rate both from the
name alone and from the summaries, and most of its gain was already there from
the name, as expected. The expectation's test for the cases the model did not
know compared each group with its own majority rate, the comparison corrected
below; for the "wrong" group, that test does not work. For the "unknown" group
it holds, on few cases.

What this run does not show:

- The name-only score is not a pure measure of memory. The parties' names
  carry information of their own, such as whether the federal government is
  involved, and a model can use that without knowing the case. Entry 02's
  planned check, a TF-IDF model on the party names alone, would show how much
  of the name-only gain that explains.
- Whether the small gain from the summaries comes from reading or from recall.
  Replacing the case and party names in the summaries and the prompt with
  neutral labels would reduce recall and help separate the two, although other
  details in a summary may still identify a well-known case.
- The "unknown" group is small (41 cases), so its accuracy is imprecise.
- It is one run of each prompt with one model. The answers can differ between
  runs, as the temperature cannot be set.
- The model gives labels rather than probabilities, so there is no AUC to
  compare with the weak ranking signal found in entries 01 and 02.

So a commercial model does beat the baseline, but most of its gain is available
from the case name alone, and this run cannot show that the summaries add to
it. As in entries 01 and 02, there is no clear evidence that the summaries carry
usable signal, although this run does not rule out a small amount. For cases
decided before a model's training cutoff, its accuracy cannot be taken at face
value. A fair test of reading would need cases decided after the cutoff (June
2026 for this model), which this dataset does not have.

## Corrections

**2026-11-24.** After an outside review of this entry:

- The split by the name-only answer compared each group with its own majority
  rate. The "correct" and "wrong" groups are defined partly by the true
  outcome, so the side that wins most often differs between them, and no single
  constant can reach both rates. In the "wrong" group, the figure given (0.709)
  was the accuracy of always answering "second", which requires knowing that
  the name-only answer was wrong. Each group is now compared with the first
  party's win rate, the accuracy of always answering "first". The earlier
  reading, that accuracy from the summary falls below the majority rate on the
  cases the model did not know and that it does worse than a constant there,
  held only for the 41 "unknown" cases and has been withdrawn.
- The comparison of the two prompts on all 638 cases, with a paired test, was
  added (`comparison.csv`), and the reading was rewritten around it. "No sign
  that the summaries themselves carry usable signal" became "no clear
  evidence": the summaries add 2.4 points over the name alone, which a paired
  test cannot tell from chance.
- The expectation above is left as it was written; its test for the "wrong"
  group used the same flawed comparison.
- A run with `--limit` no longer writes the results files, so a quick check
  cannot overwrite the figures of a full run.
