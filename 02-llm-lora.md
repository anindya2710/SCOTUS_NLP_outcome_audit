# 02 — Fine-tuned small LLM (LoRA)

**2026-11-17** (hypothesis and settings fixed before running)

**2026-11-20** (first results added)

**2026-11-23** (corrected; see Corrections at the end)

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
case here predates the model, and the summaries name the
parties, so the model may already know how famous cases ended. A gain is not
read as skill until that is checked.

## Method

Unchanged from entry 01, so the numbers are directly comparable:

- Same five stratified 80/20 splits (seeds 0–4, the same test cases), and the
  same temporal split (train on terms up to 2011, test on 2012 onward; the 57
  cases dated only by era are left out).
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
the test cases are the same ones used in entry 01. The results are in
`results_llm/`. The accuracy, AUC and truncation figures below are in those
files as written by the run, and `llm_summary.py` computes everything else
quoted below from the saved predictions (`results_llm_summary.csv` and
`results_llm_stability.csv`).

Mean of the five stratified splits, with the entry 01 models for reference:

| Model | Accuracy | ±sd | AUC | ±sd | F1 (minority) |
|---|---|---|---|---|---|
| Majority-class baseline | 0.651 | 0.000 | 0.500 | 0.000 | 0.000 |
| Random forest, full text (entry 01) | 0.638 | 0.007 | 0.573 | 0.016 | 0.196 |
| Logistic regression, full text (entry 01) | 0.648 | 0.006 | 0.569 | 0.013 | 0.095 |
| Qwen2.5-0.5B + LoRA, full text | 0.630 | 0.015 | 0.547 | 0.010 | 0.208 |
| Qwen2.5-0.5B + LoRA, outcome language removed | 0.619 | 0.029 | 0.548 | 0.025 | 0.259 |

Temporal split (train on terms up to 2011, test on the 638 cases after), all on
the full text:

| Model | Accuracy | AUC |
|---|---|---|
| Majority-class baseline | 0.671 | 0.500 |
| Random forest, TF-IDF 1,000 features (mean of five seeds) | 0.661 | 0.555 |
| Logistic regression, TF-IDF 5,000 features | 0.646 | 0.560 |
| Qwen2.5-0.5B + LoRA (one run, seed 0) | 0.660 | 0.508 |

The two TF-IDF rows come from `temporal_audit.py` (`results_temporal.csv`).
An earlier version of this table gave the forest 0.660 and labelled it full
text; that figure came from entry 01's run on the outcome + posture removed
text (see entry 01, Corrections). The LLM row is a single run, which breaks
rule 2 in `evaluation.py`; `finetune_llm.py --temporal` now runs all five
seeds.

The fine-tuned model does not beat the constant on accuracy, which was
expected. However, its AUC does not rise either. It is lower than the TF-IDF
models', at 0.547 against 0.573, and lower than the forest's on four of the
five splits. On the later terms, its ranking cannot be told apart from chance
(AUC 0.508, 95% bootstrap interval 0.46–0.55).

## Reading it

One would expect a pretrained language model to read these summaries better
than a bag of words, as it can represent negation, party roles and legal
context. In this first run, this does not prove to be the case.

The model mostly predicts the majority class. Across the five full-text
splits, it gives the first party the win in 82–90% of test cases, and catches
between 10% and 19% of the cases the first party loses. Treated as a ranking, it picks up roughly the same weak
signal as TF-IDF did: the lowest-probability fifth wins 60% of the time and
the highest fifth 72%, against 59% and 73% in entry 01.

Removing outcome language changes nothing (AUC 0.548 against 0.547), as was
the case with TF-IDF. As entry 01 now explains, most of that language
describes what the lower courts did rather than the Supreme Court's own
ruling. So this shows that the model was not drawing on it, rather than that
the text is free of leakage.

The memorization check was planned for reading a gain. As there is no gain to
explain, it matters less for this result; it is still worth running if a later
model does better.

What this run does not show:

- It is one small model (0.5 billion parameters) with one configuration, two
  epochs and no tuning. A larger model or different settings would be a new
  entry, with its own hypothesis and with any tuning done on a validation split
  carved from the training data, never on the test fold.
- The setup was simple, and some of its choices may have held the model
  back. The summaries go in as raw text with no instruction, the new
  classification head trains at the same learning rate as the adapters, the
  loss is not weighted for the 65/35 imbalance, and training stops after two
  epochs. A weak setup and a limit in the text would look the same here.
- Its predictions are also less stable than the forest's. For the 875 cases
  that fall in two or more of the five test folds, its predicted probability
  for the same case varies about twice as much between splits (median range
  0.11 against 0.05). For 1.4% of them it moves by more than 0.5, which never
  happens with the forest (`results_llm_stability.csv`). That fits an
  unstable setup at least as well as a lack of signal.
- Truncation is not the cause. At most 1.4% of summaries are longer than 512
  tokens (the count included summaries of exactly 512, so the true share is
  slightly lower).
- The training loss was printed during the run but not saved with the
  results, so I cannot yet say whether the model underfit. Training used
  float16 on the T4 GPU, and the number of steps skipped because of overflow
  was not recorded either; the learning-rate schedule kept advancing through
  any such steps. `finetune_llm.py` now saves both, and holds the schedule
  when a step is skipped.
- The temporal result is a single run (seed 0).
- One summary in the outcome-removed run was empty after stripping (ID 53693;
  its only sentence contained "held"), so the model was given an empty input
  for it.

So far, the picture from entry 01 holds: a stronger text model has not found
more signal in the summaries. This run cannot say whether that is because the
summaries carry little information about the outcome or because the setup was
too weak to find it. A better-tuned run would be needed to separate the two.

## Corrections

**2026-11-23.** After an outside review of the repository:

- The temporal table's random forest row is now the full text (it was the
  outcome + posture removed text) and a mean of five seeds, and a logistic
  regression row was added.
- The statistics quoted above (the bootstrap interval, the 82–90% and 10–19%
  ranges, and the four-of-five comparison) had no script behind them. They are
  now computed by `llm_summary.py` from the saved predictions, and all of them
  reproduce. The results are now a folder (`results_llm/`) rather than a zip
  file.
- The limits listed under "What this run does not show" were added, including
  (in a second pass the same day) how much its predictions vary between
  splits.
- `finetune_llm.py` was changed as described in its docstring. The results
  above came from the earlier version.
- The entry 01 random forest's AUC spread is now 0.016 and its minority F1
  0.196 (previously 0.015 and 0.195), after all outputs were regenerated with
  scikit-learn 1.4.2.
