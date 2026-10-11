"""Loading the Amazon Reviews 2023 data and building a leakage-safe split."""
from __future__ import annotations

import os
import urllib.request

import pandas as pd

HF_BASE = ("https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/"
           "resolve/main/raw/review_categories/{category}.jsonl")


def download(category: str = "All_Beauty", data_dir: str = "data") -> str:
    """Download one category of raw reviews (All_Beauty is ~330 MB)."""
    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, f"{category}.jsonl")
    if not os.path.exists(path):
        print(f"downloading {category} ...")
        urllib.request.urlretrieve(HF_BASE.format(category=category), path)
    return path


def load_reviews(path: str, min_reviews: int = 3, min_chars: int = 20) -> pd.DataFrame:
    """Keep users with at least `min_reviews` non-trivial reviews.

    A community graph needs users who wrote more than one review; one-off
    reviewers carry no information about who they are similar to.
    """
    df = pd.read_json(path, lines=True)
    df = df[["user_id", "parent_asin", "rating", "title", "text", "timestamp",
             "helpful_vote", "verified_purchase"]].copy()
    df["text"] = (df["title"].fillna("") + ". " + df["text"].fillna("")).str.strip()
    df["text"] = df["text"].str.replace(r"<br\s*/?>", " ", regex=True)
    df = df[df["text"].str.len() >= min_chars]
    df = df.drop_duplicates(["user_id", "parent_asin", "text"])
    counts = df["user_id"].value_counts()
    df = df[df["user_id"].isin(counts[counts >= min_reviews].index)]
    df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)
    df["review_id"] = range(len(df))
    return df


def temporal_split(df: pd.DataFrame) -> pd.DataFrame:
    """Each user's most recent review is the test review; the rest are train.

    Everything downstream (aspect profiles, the graph, communities, priors) is
    built from train rows only, so the test review never leaks into a feature.
    """
    df = df.copy()
    last = df.groupby("user_id")["timestamp"].transform("max")
    rank = df.groupby("user_id").cumcount(ascending=False)
    df["split"] = "train"
    df.loc[(df["timestamp"] == last) & (rank == 0), "split"] = "test"
    return df


def polarity_from_rating(r: pd.Series) -> pd.Series:
    """1 = positive (4-5 stars), 0 = negative (1-2 stars), NaN = neutral (3)."""
    return r.map({1: 0, 2: 0, 4: 1, 5: 1})
