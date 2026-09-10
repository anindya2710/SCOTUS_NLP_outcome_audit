"""Loading Oyez case corpus."""

import re
from pathlib import Path

import pandas as pd
DATA_PATH = Path(__file__).resolve().parent / "justice.csv"
TARGET = "first_party_winner"
TEXT = "facts"

# Language that presupposes a decision. Oyez summaries are written after the
# fact, so this can appear in text that is nominally only a statement of facts.
OUTCOME_TERMS = (
    r"\b(affirm|revers|remand|vacat|overturn|uph[eo]ld|struck down|ruled|held|decid)\w*"
)

# Procedural posture. Knowable before the ruling, but it encodes the lower
# court's disposition rather than anything about the underlying dispute, and
# the Court reverses far more often than it affirms.
PROCEDURAL_TERMS = (
    r"\b(court of appeals|circuit|district court|certiorari|summary judgment|"
    r"lower court|supreme court|appeal)\w*"
)


def load(path=DATA_PATH):
    df = pd.read_csv(path).dropna(subset=[TEXT, TARGET])
    df[TEXT] = (
        df[TEXT]
        .str.replace(r"<[^>]+>", " ", regex=True)  # Oyez ships HTML
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    return df.reset_index(drop=True)


def strip_sentences(text, patterns):
    """Drop whole sentences matching any pattern.

    Word-level removal is not enough: deleting "reversed" leaves the rest of
    "The Court of Appeals reversed the district court" carrying the same
    information.
    """
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return " ".join(
        s for s in sentences if not any(re.search(p, s, re.I) for p in patterns)
    )


def add_variants(df):
    """Attach three progressively more conservative versions of the text."""
    df = df.copy()
    df["text_full"] = df[TEXT]
    df["text_no_outcome"] = df[TEXT].apply(
        lambda t: strip_sentences(t, [OUTCOME_TERMS])
    )
    df["text_no_posture"] = df[TEXT].apply(
        lambda t: strip_sentences(t, [OUTCOME_TERMS, PROCEDURAL_TERMS])
    )
    return df


VARIANTS = {
    "text_full": "full text",
    "text_no_outcome": "outcome language removed",
    "text_no_posture": "outcome + posture removed",
}


def retention(df):
    """Fraction of characters surviving each strip, for reporting."""
    total = df["text_full"].str.len().sum()
    return {v: df[v].str.len().sum() / total for v in VARIANTS}


def labels(df):
    """Binary target. 1 = the first party (petitioner) prevailed."""
    return (df[TARGET].astype(str) == "True").astype(int).to_numpy()
