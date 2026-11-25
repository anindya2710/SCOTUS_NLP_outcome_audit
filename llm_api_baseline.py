"""Entry 03: zero-shot baseline with a commercial LLM, called through its API.

Asks Claude Haiku 5.5 which party won each case in the temporal test set (terms
after 2011: the same 638 cases as temporal_audit.py and finetune_llm.py
--temporal) and compares its accuracy with the majority rate on that set. The
model is not trained on anything here; it only reads each case and answers.

Two prompts:

facts
    The case summary and the two parties' names, close to what the other
    models see.
name
    Only the case name, term and parties: a memory probe. These cases are
    public and were decided before the model's training cutoff, so the model
    may simply remember the outcome. If it scores as well from the name alone
    as from the facts, its accuracy on the facts says little about reading them.

Setup (Windows PowerShell; on macOS or Linux use `export` instead of `$env:`):

    pip install anthropic
    $env:ANTHROPIC_API_KEY = "sk-ant-..."      # never put the key in a file in the repo
    python llm_api_baseline.py --prompt facts --limit 20    # a cheap first check
    python llm_api_baseline.py --prompt facts
    python llm_api_baseline.py --prompt name

Answers are saved to results_llm_api/<prompt>.jsonl as they arrive, so a run
that stops part-way resumes where it left off and no case is paid for twice.
Each run updates results_llm_api/summary.csv. Once both prompts have been run,
it also writes results_llm_api/memory_split.csv: accuracy from the summaries,
split by whether the name-only answer for the same case was correct, wrong or
unknown.
"""

import argparse
import json
import re
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

import data

MODEL = "claude-haiku-5-5"
OUT_DIR = Path(__file__).resolve().parent / "results_llm_api"

SYSTEM = (
    "You predict the outcomes of U.S. Supreme Court cases. "
    "Reply with a single word and nothing else."
)

PROMPTS = {
    "facts": (
        "Case summary:\n{facts}\n\n"
        "First party (petitioner): {first}\n"
        "Second party (respondent): {second}\n\n"
        "Which party did the Supreme Court rule for? Reply first or second."
    ),
    "name": (
        "Case: {name} (term {term})\n"
        "First party (petitioner): {first}\n"
        "Second party (respondent): {second}\n\n"
        "From what you know of this case, which party did the Supreme Court rule "
        "for? Reply first or second, or unknown if you do not know the case."
    ),
}

ANSWER = re.compile(r"\b(first|second|unknown)\b", re.I)


def temporal_test_set(df, quantile=0.8):
    """Same split as evaluation.temporal_evaluate: test on the terms after the
    80th-percentile term (2011). Cases dated only by era are left out."""
    term = data.term_year(df)
    keep = term.notna()
    cutoff = term[keep].quantile(quantile)
    return df[keep & (term > cutoff)], int(cutoff)


def parse(text):
    """'first', 'second', 'unknown', or 'unparsed'."""
    m = ANSWER.search(text or "")
    return m.group(1).lower() if m else "unparsed"


def ask(client, model, prompt_text):
    response = client.messages.create(
        model=model,
        max_tokens=10,
        system=SYSTEM,
        messages=[{"role": "user", "content": prompt_text}],
        # A one-word answer needs no extended reasoning, and thinking tokens
        # would be billed as output.
        thinking={"type": "disabled"},
        output_config={"effort": "low"},
    )
    text = "".join(b.text for b in response.content if b.type == "text").strip()
    return text, response


def load_done(path):
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return {r["ID"]: r for r in rows}


def summarize(rows, y_by_id, prompt):
    ids = list(rows)
    answers = np.array([rows[i]["answer"] for i in ids])
    y = np.array([y_by_id[i] for i in ids])
    answered = np.isin(answers, ["first", "second"])
    pred = (answers[answered] == "first").astype(int)
    y_ans = y[answered]
    return {
        "prompt": prompt,
        "model": rows[ids[0]]["model"],
        "n_cases": len(ids),
        "majority_rate_all": float(max(y.mean(), 1 - y.mean())),
        "n_answered": int(answered.sum()),
        "n_unknown": int((answers == "unknown").sum()),
        "n_unparsed": int((answers == "unparsed").sum()),
        "accuracy_answered": float((pred == y_ans).mean()) if answered.any() else float("nan"),
        "majority_rate_answered": float(max(y_ans.mean(), 1 - y_ans.mean())) if answered.any() else float("nan"),
        "predicted_first_share": float(pred.mean()) if answered.any() else float("nan"),
        "input_tokens": int(sum(rows[i]["input_tokens"] for i in ids)),
        "output_tokens": int(sum(rows[i]["output_tokens"] for i in ids)),
    }


def memory_split(facts_rows, name_rows, y_by_id):
    """Accuracy from the summaries, grouped by the name-only answer.

    If the model does well from the summaries only where it already knew the
    winner from the name, its accuracy comes from memory, not from reading.
    """
    groups = {"name correct": [], "name wrong": [], "name unknown": []}
    for i, y in y_by_id.items():
        if i not in facts_rows or i not in name_rows:
            continue
        guess = name_rows[i]["answer"]
        if guess in ("first", "second"):
            key = "name correct" if (guess == "first") == bool(y) else "name wrong"
        else:
            key = "name unknown"            # 'unknown' or an unreadable reply
        groups[key].append((facts_rows[i]["answer"], y))

    out = []
    for key, pairs in groups.items():
        answered = [(a, y) for a, y in pairs if a in ("first", "second")]
        ys = np.array([y for _, y in answered])
        out.append({
            "group": key,
            "n_cases": len(pairs),
            "n_facts_answered": len(answered),
            "facts_accuracy": float(np.mean([(a == "first") == bool(y) for a, y in answered]))
                              if answered else float("nan"),
            "majority_rate": float(max(ys.mean(), 1 - ys.mean())) if answered else float("nan"),
        })
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--prompt", choices=list(PROMPTS), default="facts")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--limit", type=int, default=None,
                    help="only the first N test cases, for a cheap check")
    args = ap.parse_args()

    import anthropic  # imported here so the rest of the file loads without it
    client = anthropic.Anthropic(max_retries=5)   # reads ANTHROPIC_API_KEY

    df = data.load(strict=True)                   # decodes HTML entities
    test, cutoff = temporal_test_set(df)
    if args.limit:
        test = test.head(args.limit)
    y_by_id = dict(zip(test["ID"], data.labels(test)))
    print(f"{len(test)} test cases (terms after {cutoff}), prompt '{args.prompt}', model {args.model}")

    OUT_DIR.mkdir(exist_ok=True)
    path = OUT_DIR / f"{args.prompt}.jsonl"
    done = load_done(path)
    todo = test[~test["ID"].isin(done)]
    print(f"{len(done)} already answered, {len(todo)} to ask")

    with path.open("a", encoding="utf-8") as f:
        for k, row in enumerate(todo.itertuples(index=False), 1):
            prompt_text = PROMPTS[args.prompt].format(
                facts=row.facts, name=row.name, term=row.term,
                first=row.first_party, second=row.second_party,
            )
            text, response = ask(client, args.model, prompt_text)
            record = {
                "ID": row.ID, "answer": parse(text), "raw": text,
                "stop_reason": response.stop_reason,
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "model": response.model, "date": date.today().isoformat(),
            }
            f.write(json.dumps(record) + "\n")
            f.flush()
            done[row.ID] = record
            if k % 50 == 0:
                print(f"  {k}/{len(todo)}")

    rows = {i: done[i] for i in test["ID"] if i in done}
    s = summarize(rows, y_by_id, args.prompt)
    for key, value in s.items():
        print(f"{key:>24}: {value:.3f}" if isinstance(value, float) else f"{key:>24}: {value}")

    summary_path = OUT_DIR / "summary.csv"
    old = pd.read_csv(summary_path) if summary_path.exists() else pd.DataFrame()
    if not old.empty:
        old = old[old["prompt"] != args.prompt]
    pd.concat([old, pd.DataFrame([s])]).to_csv(summary_path, index=False)

    facts_done = load_done(OUT_DIR / "facts.jsonl")
    name_done = load_done(OUT_DIR / "name.jsonl")
    if facts_done and name_done:
        split = memory_split(facts_done, name_done, y_by_id)
        print("\nAccuracy from the summaries, by the name-only answer:")
        print(split.to_string(index=False, float_format="%.3f"))
        split.to_csv(OUT_DIR / "memory_split.csv", index=False)


if __name__ == "__main__":
    main()
