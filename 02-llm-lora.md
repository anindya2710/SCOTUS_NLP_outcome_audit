# 02 — Fine-tuned small LLM (LoRA)

**2026-11-17** (hypothesis and settings fixed before running)

**2026-11-20** (first results added)

## Question

Does a small open language model, fine-tuned with LoRA on the Oyez facts
summaries, beat the fixed baseline of 0.651 accuracy and 0.500 AUC, or the
TF-IDF models' 0.573 AUC from entry 01?

Entry 01 planned a frozen transformer encoder as the next step. I have moved
LoRA fine-tuning ahead of it, as fine-tuning asks the same question (does a
better representation of the text help?) with a model that also adapts to the
task. The frozen encoder is now entry 03, and the reframed target is entry 04.
Entry 01 is left as it was written.

## Expectation before running

AUC rises above the TF-IDF models' 0.573, to roughly 0.60–0.65, because a
pretrained language model can represent negation, party roles and legal
context that bag-of-words cannot. Accuracy probably still does not clear
0.651, for the same reason as in entry 01: with a 65/35 target, a model has to
be confidently right about the minority class to beat a constant.

Some of any gain may come from pretraining rather than from the facts. Every
case here (terms 1955–2020) predates the model, and the summaries name the
parties, so the model may already know how famous cases ended. A gain is not
read as skill until that is checked.

## Method

Unchanged from entry 01, so the numbers are directly comparable:

- Same five stratified 80/20 splits (seeds 0–4, the same test cases), and the
  same temporal split (train on terms up to 2011, test on 2012 onward).
- Same fixed baseline: 0.651 accuracy, 0.500 AUC.
- Accuracy reported next to the majority rate, with AUC, minority-class F1,
  and the win rates in the lowest and highest predicted-probability fifths.

Model and training, fixed before running (`CONFIG` in `finetune_llm.py`):

- `Qwen/Qwen2.5-0.5B` (base model), with a two-class classification head.
- LoRA adapters (r = 16, alpha = 32, dropout 0.05) on the attention
  projections; the classification head trained in full.
- Two epochs, learning rate 2e-4 with linear warmup over 10% of steps and
  linear decay, effective batch size 16, summaries truncated at 512 tokens.
- No early stopping and no hyperparameter search, because the only held-out
  data is the test fold.
- Decision threshold 0.5, as in entry 01.

Runs: full text, then outcome language removed (the entry 01 variants), then
the temporal split.

Planned check before reading any gain as skill: compare the model with TF-IDF
on the party names alone, and the full-text model with and without the party
names masked. A language model that does much better than TF-IDF from names
alone is drawing on what it remembers.

## Result

These are first results, from the single configuration fixed above. Each run
took about six minutes on a Kaggle T4 GPU. The prediction files confirm that
the test cases are the same ones used in entry 01.

Mean of the five stratified splits, with the entry 01 models for reference:

| Model | Accuracy | ±sd | AUC | ±sd | F1 (minority) |
|---|---|---|---|---|---|
| Majority-class baseline | 0.651 | 0.000 | 0.500 | 0.000 | 0.000 |
| Random forest, full text (entry 01) | 0.638 | 0.007 | 0.573 | 0.015 | 0.195 |
| Logistic regression, full text (entry 01) | 0.648 | 0.006 | 0.569 | 0.013 | 0.095 |
| Qwen2.5-0.5B + LoRA, full text | 0.630 | 0.015 | 0.547 | 0.010 | 0.208 |
| Qwen2.5-0.5B + LoRA, outcome language removed | 0.619 | 0.029 | 0.548 | 0.025 | 0.259 |

Temporal split (train on terms up to 2011, test on the 638 cases after):

Temporal split (train on terms up to 2011, test on the 638 cases after), all on
the full text:

| Model | Accuracy | AUC |
|---|---|---|
| Majority-class baseline | 0.671 | 0.500 |
| Random forest, TF-IDF 1,000 features | 0.665 | 0.553 |
| Logistic regression, TF-IDF 5,000 features | 0.646 | 0.560 |
| Qwen2.5-0.5B + LoRA | 0.660 | 0.508 |

The two TF-IDF rows come from `temporal_audit.py` (`results_temporal.csv`),
which runs the temporal split for every text version and both models. An
earlier version of this table labelled the forest's accuracy as full text,
but that 0.660 came from entry 01's run on the outcome + posture removed text.
The fine-tuned model does not beat the constant on accuracy, which was
expected. However, its AUC does not rise either. It is lower than the TF-IDF
models', at 0.547 against 0.573, and lower than the forest's on four of the
five splits. On the later terms, its ranking cannot be told apart from chance
(AUC 0.508, 95% bootstrap interval 0.46–0.55).

## Reading it

One would expect a pretrained language model to read these summaries better
than a bag of words, as it can represent negation, party roles and legal
context. In this first run, this does not prove to be the case.

The model mostly predicts the majority class. It gives the first party the win
in 82–90% of test cases, and catches between 10% and 19% of the cases the
first party loses. Treated as a ranking, it picks up roughly the same weak
signal as TF-IDF did: the lowest-probability fifth wins 60% of the time and
the highest fifth 72%, against 59% and 73% in entry 01.

Removing outcome language changes nothing (AUC 0.548 against 0.547), as was
the case with TF-IDF. Thus, there is no leakage advantage here either.

The memorization check was planned for reading a gain. As there is no gain to
explain, it matters less for this result; it is still worth running if a later
model does better.

What this run does not show:

- It is one small model (0.5 billion parameters) with one configuration, two
  epochs and no tuning. A larger model or different settings would be a new
  entry, with its own hypothesis and with any tuning done on a validation split
  carved from the training data, never on the test fold.
- Truncation is not the cause. Only 1.4% of summaries are longer than 512
  tokens.
- The training loss was printed during the run but not saved with the
  results, so I cannot yet say whether the model underfit. Future runs will
  save it.

So far, the picture from entry 01 holds: the summaries appear to carry very
little information about the outcome, and a stronger text model has not
changed that. One run of one small model is not enough to settle it, however.
