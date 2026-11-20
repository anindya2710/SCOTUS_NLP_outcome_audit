"""Entry 02: LoRA fine-tuning of a small open LLM.

The protocol is the one from entry 01, unchanged: the same five stratified
80/20 splits (same seeds, so the same test cases), the same temporal split, and
the same fixed baseline of 0.651 accuracy and 0.500 AUC. Only the model changes.

Hyperparameters are fixed in CONFIG before running and are not tuned on the
test folds. There is no early stopping, because the only held-out data is the
test fold, and choosing an epoch by looking at it would leak.

Usage (on a GPU):
    python finetune_llm.py                      # five stratified splits
    python finetune_llm.py --temporal           # train <= 2011, test after
    python finetune_llm.py --variant text_no_outcome
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    set_seed,
)

import data
from evaluation import SEEDS, majority_rate, split

# Fixed before running. Changing any of these is a new entry, not a re-run.
CONFIG = {
    "model": "Qwen/Qwen2.5-0.5B",  # base model, not the chat version
    "max_length": 512,             # tokens; longer summaries are truncated
    "epochs": 2,
    "batch_size": 8,
    "grad_accum": 2,               # effective batch of 16
    "lr": 2e-4,
    "warmup_ratio": 0.1,
    "weight_decay": 0.0,
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    # LoRA adapters on the attention projections only. The classification
    # head is new, so PEFT trains it in full (TaskType.SEQ_CLS does this).
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
}


def temporal_indices(df, quantile=0.8):
    """Same split as evaluation.temporal_evaluate: train on terms up to the
    80th-percentile term (2011), test on the terms after it."""
    term = pd.to_numeric(df["term"], errors="coerce")
    keep = term.notna().to_numpy()
    cutoff = term[keep].quantile(quantile)
    tr = np.flatnonzero(keep & (term <= cutoff).to_numpy())
    te = np.flatnonzero(keep & (term > cutoff).to_numpy())
    return tr, te, int(cutoff)


def load_model(cfg, device):
    tok = AutoTokenizer.from_pretrained(cfg["model"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    model = AutoModelForSequenceClassification.from_pretrained(cfg["model"], num_labels=2)
    # Keep the weights in float32. Recent versions of transformers load Qwen in
    # bfloat16 by default, which the T4 GPU does not support and which the
    # float16 gradient scaler cannot handle. Mixed precision comes from autocast.
    model = model.float()
    # The classifier reads the hidden state of the last non-padding token,
    # which it finds through pad_token_id.
    model.config.pad_token_id = tok.pad_token_id

    lora = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        target_modules=cfg["target_modules"],
    )
    model = get_peft_model(model, lora).to(device)
    model.print_trainable_parameters()
    return tok, model


def batches(enc, idx, batch_size, tok, labels=None):
    for start in range(0, len(idx), batch_size):
        chunk = idx[start:start + batch_size]
        batch = tok.pad(
            {
                "input_ids": [enc["input_ids"][i] for i in chunk],
                "attention_mask": [enc["attention_mask"][i] for i in chunk],
            },
            return_tensors="pt",
        )
        if labels is not None:
            batch["labels"] = torch.tensor(labels[chunk], dtype=torch.long)
        yield chunk, batch


def fit_and_predict(texts, y, tr, te, seed, cfg, device):
    """Train on tr, return P(first party wins) for te."""
    set_seed(seed)
    tok, model = load_model(cfg, device)
    enc = tok(list(texts), truncation=True, max_length=cfg["max_length"])

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=cfg["lr"], weight_decay=cfg["weight_decay"])

    steps_per_epoch = math.ceil(math.ceil(len(tr) / cfg["batch_size"]) / cfg["grad_accum"])
    total_steps = steps_per_epoch * cfg["epochs"]
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(cfg["warmup_ratio"] * total_steps), total_steps
    )

    rng = np.random.default_rng(seed)
    model.train()
    for epoch in range(cfg["epochs"]):
        order = rng.permutation(tr)
        running, n = 0.0, 0
        for step, (_, batch) in enumerate(batches(enc, order, cfg["batch_size"], tok, y)):
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("labels")
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(**batch).logits
            # Loss in float32 for stability under mixed precision.
            loss = F.cross_entropy(logits.float(), labels) / cfg["grad_accum"]
            scaler.scale(loss).backward()
            running += loss.item() * cfg["grad_accum"]
            n += 1
            last = step + 1 == math.ceil(len(order) / cfg["batch_size"])
            if (step + 1) % cfg["grad_accum"] == 0 or last:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
        print(f"  epoch {epoch + 1}: mean training loss {running / n:.4f}", flush=True)

    # Score the test fold, longest first to keep padding low, then restore order.
    model.eval()
    lengths = np.array([len(enc["input_ids"][i]) for i in te])
    by_length = te[np.argsort(-lengths)]
    proba = {}
    with torch.no_grad():
        for chunk, batch in batches(enc, by_length, cfg["batch_size"] * 2, tok):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(**batch).logits
            p = torch.softmax(logits.float(), dim=-1)[:, 1].cpu().numpy()
            proba.update(zip(chunk, p))

    truncated = float(np.mean([len(enc["input_ids"][i]) >= cfg["max_length"] for i in range(len(texts))]))
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return np.array([proba[i] for i in te]), truncated


def score(y_true, proba):
    pred = (proba >= 0.5).astype(int)
    order = np.argsort(proba)
    fifth = len(order) // 5
    return {
        "accuracy": accuracy_score(y_true, pred),
        "auc": roc_auc_score(y_true, proba),
        "f1_minority": f1_score(y_true, pred, pos_label=0),  # minority class
        "majority_rate": majority_rate(y_true),
        "bottom_quintile_win_rate": y_true[order[:fifth]].mean(),
        "top_quintile_win_rate": y_true[order[-fifth:]].mean(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="text_full", choices=list(data.VARIANTS))
    ap.add_argument("--temporal", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--model", default=None, help="override CONFIG['model']")
    ap.add_argument("--epochs", type=int, default=None, help="override, for smoke tests only")
    ap.add_argument("--limit", type=int, default=None, help="use the first N cases, for smoke tests only")
    ap.add_argument("--out", default="results_llm")
    args = ap.parse_args()

    cfg = dict(CONFIG)
    if args.model:
        cfg["model"] = args.model
    if args.epochs:
        cfg["epochs"] = args.epochs

    df = data.add_variants(data.load())
    if args.limit:
        df = df.iloc[: args.limit].reset_index(drop=True)
    texts = df[args.variant].to_numpy()
    y = data.labels(df)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    tag = f"{args.variant}_{'temporal' if args.temporal else 'stratified'}"
    print(f"{tag} on {device}, model {cfg['model']}, {len(df)} cases", flush=True)

    if args.temporal:
        tr, te, cutoff = temporal_indices(df)
        runs = [("temporal", 0, tr, te)]
        print(f"temporal split: train terms <= {cutoff} (n={len(tr)}), test after (n={len(te)})")
    else:
        runs = [(f"seed {s}", s, *split(len(texts), y, s)) for s in args.seeds]

    rows, preds = [], []
    for name, seed, tr, te in runs:
        t0 = time.time()
        print(f"{name}: training on {len(tr)}, testing on {len(te)}", flush=True)
        proba, truncated = fit_and_predict(texts, y, np.asarray(tr), np.asarray(te), seed, cfg, device)
        m = score(y[te], proba)
        m.update(run=name, seed=seed, n_train=len(tr), n_test=len(te),
                 truncated_share=truncated, minutes=(time.time() - t0) / 60)
        rows.append(m)
        preds.append(pd.DataFrame({"run": name, "ID": df["ID"].to_numpy()[te],
                                   "y": y[te], "p_first_party_wins": proba}))
        print(f"  accuracy {m['accuracy']:.3f} (majority {m['majority_rate']:.3f}), "
              f"AUC {m['auc']:.3f}, minority F1 {m['f1_minority']:.3f}, "
              f"{m['minutes']:.1f} min", flush=True)

    runs_df = pd.DataFrame(rows)
    runs_df.to_csv(out / f"runs_{tag}.csv", index=False)
    pd.concat(preds).to_csv(out / f"predictions_{tag}.csv", index=False)
    summary = {
        "variant": args.variant,
        "split": "temporal" if args.temporal else "stratified",
        "accuracy": runs_df["accuracy"].mean(),
        "accuracy_sd": runs_df["accuracy"].std(ddof=0),
        "auc": runs_df["auc"].mean(),
        "auc_sd": runs_df["auc"].std(ddof=0),
        "f1_minority": runs_df["f1_minority"].mean(),
        "majority_rate": runs_df["majority_rate"].mean(),
        "config": cfg,
    }
    (out / f"summary_{tag}.json").write_text(json.dumps(summary, indent=2, default=float))
    print(json.dumps({k: v for k, v in summary.items() if k != "config"}, indent=2, default=float))


if __name__ == "__main__":
    main()
