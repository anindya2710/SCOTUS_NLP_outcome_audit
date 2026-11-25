"""Loading the Oyez case corpus and building the text versions.

There are two cleaning modes. The default is the one entries 01 and 02 were
run with, kept unchanged so that their numbers can be reproduced.
`strict=True` fixes three problems found in an outside review (2026-11-23):

- HTML entities such as `&amp;` were not decoded, so `amp` entered the
  vocabulary from 110 summaries.
- The word patterns matched more than intended. `affirm\\w*` caught
  "affirmative action", `vacat\\w*` caught "vacation", and `held` and `decid`
  caught everyday uses such as "held in jail" and "decided to".
- Sentences were split at every full stop, including the ones in "v." and
  "U.S.", so removing a sentence could leave a fragment of it behind.

supporting_checks.py reruns the entry 01 models on the strict versions.
"""

import html
import re
from pathlib import Path

import pandas as pd

DATA_PATH = Path(__file__).resolve().parent / "justice.csv"
TARGET = "first_party_winner"
TEXT = "facts"

# Language describing a ruling. Most matching sentences describe a LOWER
# court's ruling ("The Court of Appeals affirmed"), which is known before the
# Supreme Court decides and so is not leakage. Removing them still sets a
# bound: any wording that does give away the Supreme Court's own ruling also
# matches, so if removing all of it changes nothing, that wording was not
# being used.
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

# Strict versions. Whole words only, and words with everyday meanings
# ("held", "decided", "ruled") count only in a sentence that also names a
# court, a judge or a jury.
STRICT_OUTCOME_TERMS = (
    r"\b(affirm(ed|s|ing|ance)?|revers(ed|al|ing)|remand(ed|s|ing)?|"
    r"vacat(e|ed|es|ing|ur)|overturn(ed|s|ing)?|uph[eo]ld(s|ing)?|struck down)\b"
)
STRICT_RULING_WORDS = r"\b(ruled|ruling|held|holding|decid(ed|es|ing)|decision)\b"
COURT_WORDS = r"\b(courts?|judges?|jury|panel|justices?|board|commission|tribunal)\b"
STRICT_PROCEDURAL_TERMS = (
    r"\b(court of appeals|circuits?|district court|certiorari|summary judgment|"
    r"lower court|supreme court|appeal(s|ed|ing)?)\b"
)

# Full stops that do not end a sentence. Errs toward longer sentences: "U.S."
# at the end of a sentence joins it to the next one. Longer forms come first
# ("U.S.C" before "U.S"), because the first alternative that matches wins.
ABBREVIATIONS = (
    r"\b(?:v|vs|U\.S\.C|U\.S|Mr|Ms|Mrs|Dr|Co|Corp|Inc|Ltd|St|Jr|Sr|No|Nos|"
    r"Stat|Fed|App|Cir|Ct|Art|Sec|Gen|Gov|Rep|Sen|Lt|Sgt|Capt|Col|Mt|Ft|al|etc|"
    r"e\.g|i\.e|Cal|Ill|Pa|Tex|Fla|Mass|Mich|Wash|Va|Ga|Ala|Miss|La|Ky|Tenn|"
    r"Md|Conn|Ariz|Colo|Okla|Wis|Minn|Mo|Ind|Neb|Kan|Ore|Nev)\."
)


def load(path=DATA_PATH, strict=False):
    df = pd.read_csv(path).dropna(subset=[TEXT, TARGET])
    text = df[TEXT]
    if strict:
        # Decodes &amp; and also turns &lt;i&gt; into <i>, which the next
        # step removes.
        text = text.map(html.unescape)
    df[TEXT] = (
        text
        .str.replace(r"<[^>]+>", " ", regex=True)  # Oyez ships HTML
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    return df.reset_index(drop=True)


def term_year(df):
    """The term as a number.

    57 cases from before 1955 have their term recorded as an era ("1900-1940")
    rather than a year. They come back as NaN, so the temporal split leaves
    them out; the stratified splits include them.
    """
    return pd.to_numeric(df["term"], errors="coerce")


def split_sentences(text, strict=False):
    if not strict:
        return re.split(r"(?<=[.!?])\s+", text)
    protected = re.sub(ABBREVIATIONS, lambda m: m.group(0).replace(".", "\0"), text)
    return [s.replace("\0", ".") for s in re.split(r"(?<=[.!?])\s+", protected)]


def is_strict_outcome(sentence):
    return bool(
        re.search(STRICT_OUTCOME_TERMS, sentence, re.I)
        or (re.search(STRICT_RULING_WORDS, sentence, re.I)
            and re.search(COURT_WORDS, sentence, re.I))
    )


def _matches(pattern, sentence):
    if callable(pattern):
        return pattern(sentence)
    return re.search(pattern, sentence, re.I) is not None


def strip_sentences(text, patterns, strict=False):
    """Drop whole sentences matching any pattern (a regex or a function).

    Word-level removal is not enough: deleting "reversed" leaves the rest of
    "The Court of Appeals reversed the district court" carrying the same
    information.
    """
    return " ".join(
        s for s in split_sentences(text, strict)
        if not any(_matches(p, s) for p in patterns)
    )


def add_variants(df, strict=False):
    """Attach three progressively more conservative versions of the text.

    A few summaries lose every sentence and become empty strings (1 and 13 in
    the default mode, 4 and 25 in the strict one).
    """
    df = df.copy()
    outcome = [is_strict_outcome] if strict else [OUTCOME_TERMS]
    posture = [STRICT_PROCEDURAL_TERMS] if strict else [PROCEDURAL_TERMS]
    df["text_full"] = df[TEXT]
    df["text_no_outcome"] = df[TEXT].apply(
        lambda t: strip_sentences(t, outcome, strict)
    )
    df["text_no_posture"] = df[TEXT].apply(
        lambda t: strip_sentences(t, outcome + posture, strict)
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
