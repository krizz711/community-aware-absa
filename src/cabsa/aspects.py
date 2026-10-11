"""Aspect extraction: a seed lexicon of aspect categories, matched per clause.

Reviews are split into sentences and then into clauses at contrast markers
("but", "however", ...), so that "smells great but the pump broke" yields a
positive SCENT mention and a negative PACKAGING mention instead of one mixed
sentence. The lexicon is a starting point; `mine_candidate_terms` lists
frequent noun phrases the lexicon does not cover yet, which is how it was
expanded.
"""
from __future__ import annotations

import re
from collections import Counter

import pandas as pd

ASPECT_LEXICON: dict[str, list[str]] = {
    "scent": ["scent", "scents", "smell", "smells", "smelled", "smelling", "fragrance",
              "perfume", "odor", "odour", "aroma"],
    "texture": ["texture", "consistency", "greasy", "sticky", "oily", "creamy", "lather",
                "foam", "foamy", "absorbs", "absorb", "residue", "lightweight", "runny"],
    "price": ["price", "priced", "value", "money", "cost", "expensive", "cheap", "cheaper",
              "worth", "deal", "overpriced", "affordable", "bargain"],
    "packaging": ["packaging", "package", "bottle", "container", "pump", "cap", "lid",
                  "tube", "jar", "nozzle", "dispenser", "leaked", "leaking", "leaks", "seal"],
    "skin_effect": ["skin", "face", "acne", "breakout", "breakouts", "rash", "irritation",
                    "irritated", "moisturizing", "moisturized", "hydrating", "pores",
                    "wrinkles", "redness", "itchy", "burning", "allergic"],
    "hair_effect": ["hair", "scalp", "curls", "frizz", "frizzy", "dandruff", "split ends",
                    "shiny", "volume"],
    "longevity": ["lasts", "lasting", "long-lasting", "long lasting", "durable", "wears off",
                  "stays on", "fades", "faded", "all day"],
    "color": ["color", "colour", "colors", "shade", "shades", "tint", "pigment",
              "pigmented"],
    "size": ["size", "amount", "quantity", "ounce", "ounces", "oz", "tiny", "travel size",
             "smaller than", "bigger than"],
    "delivery": ["shipping", "delivery", "delivered", "arrived", "seller", "shipped",
                 "damaged", "expired", "counterfeit", "fake"],
    "effectiveness": ["works", "worked", "effective", "results", "difference", "useless",
                      "does nothing", "did nothing", "noticeable"],
    "application": ["apply", "applies", "application", "applicator", "brush", "easy to use",
                    "blend", "blends", "spreads", "instructions"],
}

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
_CLAUSE_SPLIT = re.compile(r"\b(?:but|however|although|though|except|yet|whereas)\b|;",
                           re.IGNORECASE)


def _compile(lexicon: dict[str, list[str]]) -> dict[str, re.Pattern]:
    return {a: re.compile(r"\b(?:" + "|".join(re.escape(t) for t in sorted(terms, key=len, reverse=True)) + r")\b",
                          re.IGNORECASE)
            for a, terms in lexicon.items()}


def split_clauses(text: str) -> list[str]:
    clauses = []
    for sent in _SENT_SPLIT.split(text):
        for c in _CLAUSE_SPLIT.split(sent):
            c = c.strip(" ,.-")
            if len(c) > 2:
                clauses.append(c)
    return clauses


def extract_mentions(df: pd.DataFrame, lexicon: dict[str, list[str]] | None = None) -> pd.DataFrame:
    """One row per (review, clause, aspect) where an aspect term occurs."""
    patterns = _compile(lexicon or ASPECT_LEXICON)
    rows = []
    for rid, text in zip(df["review_id"], df["text"]):
        for ci, clause in enumerate(split_clauses(text)):
            for aspect, pat in patterns.items():
                m = pat.search(clause)
                if m:
                    rows.append((rid, ci, aspect, m.group(0).lower(), clause))
    return pd.DataFrame(rows, columns=["review_id", "clause_idx", "aspect", "term", "clause"])


def mine_candidate_terms(texts, top_n: int = 60, n_process: int = 1) -> pd.DataFrame:
    """Frequent noun-phrase heads (spaCy) that are not in the lexicon yet."""
    import spacy

    nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
    known = {t for terms in ASPECT_LEXICON.values() for t in terms}
    counts: Counter = Counter()
    for doc in nlp.pipe(texts, batch_size=256, n_process=n_process):
        for chunk in doc.noun_chunks:
            head = chunk.root.text.lower()
            if chunk.root.pos_ == "NOUN" and head.isalpha() and head not in known:
                counts[head] += 1
    return pd.DataFrame(counts.most_common(top_n), columns=["term", "count"])
