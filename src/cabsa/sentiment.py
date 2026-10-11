"""Aspect-level sentiment for each extracted mention.

Default scorer is VADER on the clause that holds the aspect term (fast, runs
on CPU, no training). `score_transformer` is the stronger alternative: a
DeBERTa model fine-tuned for aspect-based sentiment, which reads the whole
sentence together with the aspect term. Use it on a GPU runtime.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

POS, NEG = 0.05, -0.05


def score_vader(mentions: pd.DataFrame) -> pd.DataFrame:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    sia = SentimentIntensityAnalyzer()
    cache: dict[str, float] = {}
    scores = []
    for clause in mentions["clause"]:
        if clause not in cache:
            cache[clause] = sia.polarity_scores(clause)["compound"]
        scores.append(cache[clause])
    out = mentions.copy()
    out["score"] = scores
    out["polarity"] = np.select([out["score"] > POS, out["score"] < NEG], [1, -1], 0)
    return out


def score_transformer(mentions: pd.DataFrame,
                      model_name: str = "yangheng/deberta-v3-base-absa-v1.1",
                      batch_size: int = 64, device: int | None = None) -> pd.DataFrame:
    """Score (clause, aspect-term) pairs with a pretrained ABSA classifier."""
    import torch
    from transformers import pipeline

    if device is None:
        device = 0 if torch.cuda.is_available() else -1
    clf = pipeline("text-classification", model=model_name, device=device, top_k=None)
    inputs = [{"text": c, "text_pair": t} for c, t in zip(mentions["clause"], mentions["term"])]
    preds = clf(inputs, batch_size=batch_size, truncation=True, max_length=128)
    score = []
    for p in preds:
        d = {x["label"].lower(): x["score"] for x in p}
        score.append(d.get("positive", 0.0) - d.get("negative", 0.0))
    out = mentions.copy()
    out["score"] = score
    out["polarity"] = np.select([out["score"] > 0.2, out["score"] < -0.2], [1, -1], 0)
    return out


def review_level_check(mentions: pd.DataFrame, reviews: pd.DataFrame) -> pd.DataFrame:
    """Sanity check: mean aspect sentiment per star rating (should rise with stars)."""
    per_review = mentions.groupby("review_id")["score"].mean().rename("aspect_score")
    j = reviews.set_index("review_id")[["rating"]].join(per_review, how="inner")
    return j.groupby("rating")["aspect_score"].agg(["mean", "std", "count"])
