# 03 — Zero-shot commercial LLM through its API

**2026-11-24** (question, expectation and settings fixed before the full runs)

**2026-11-24** (results added)

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
below are in `summary.csv` and `memory_split.csv` in the same folder.

| Prompt | Cases answered | Accuracy | Majority rate (same cases) | Share given to the first party |
|---|---|---|---|---|
| Summary and parties (`facts`) | 638 of 638 | 0.759 | 0.671 | 0.652 |
| Case name, term and parties only (`name`) | 597 of 638 | 0.747 | 0.678 | 0.784 |

From the name alone, the model answered "unknown" for 39 cases, and 2 of its
replies could not be read; both are counted as unknown below.

Accuracy from the summaries, split by the name-only answer for the same case:

| Name-only answer | Cases | Accuracy from the summary | Majority rate |
|---|---|---|---|
| Correct | 446 | 0.843 | 0.809 |
| Wrong | 151 | 0.576 | 0.709 |
| Unknown | 41 | 0.512 | 0.561 |

Both runs together used 485,197 input tokens and 4,292 output tokens, about
$0.05 at the listed prices.

## Reading it

This is the first model in this log to clearly beat the majority rate: from the
summaries, it is right on 0.759 of the later cases, against 0.671 for a
constant prediction. However, it does almost as well without the summaries.
Given only the case name, term and parties, it names the winner correctly in
446 of the 638 cases, and reaches 0.747 on the 597 it answers. So most of what
it gets right does not need the summary.

The split points the same way. Where the name-only answer was correct, the
answer from the summary is right 84% of the time, about 3 points above the
majority rate of those cases. Where the name-only answer was wrong or unknown,
accuracy from the summary falls below the majority rate (0.576 against 0.709,
and 0.512 against 0.561). On the cases the model does not appear to know, it
does worse than a constant. This is what I expected, and it is in line with
entries 01 and 02: there is still no sign that the summaries themselves carry
usable signal.

What this run does not show:

- The name-only score is not a pure measure of memory. The parties' names
  carry information of their own, such as whether the federal government is
  involved, and a model can use that without knowing the case. Entry 02's
  planned check, a TF-IDF model on the party names alone, would show how much
  of the 0.747 that explains.
- The split cannot fully separate memory from the model's own lean. Both
  prompts go to the same model, so a case it misjudges from the name it may
  misjudge from the summary in the same way, whether or not it remembers the
  case. The "wrong" group is, by construction, a harder set of cases.
- The "unknown" group is small (41 cases), so its accuracy is imprecise.
- It is one run of each prompt with one model. The answers can differ between
  runs, as the temperature cannot be set.
- The model gives labels rather than probabilities, so there is no AUC to
  compare with the weak ranking signal found in entries 01 and 02.

So a commercial model does beat the baseline, but mostly with what it already
knows about the cases rather than with what it reads in the summaries. For
cases decided before a model's training cutoff, its accuracy cannot be taken
at face value. A fair test of reading would need cases decided after the
cutoff (June 2026 for this model), which this dataset does not have.
