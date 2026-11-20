# 02 — Fine-tuned small LLM (LoRA)

**2026-11-17** (hypothesis and settings fixed before running; results pending)

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

*To be filled in after running.*

## Reading it

*To be filled in after running.*
