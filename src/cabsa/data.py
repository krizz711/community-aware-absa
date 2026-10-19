"""Loading the Amazon Reviews 2023 data and building a leakage-safe temporal split."""
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


def load_raw(path: str, min_chars: int = 20) -> pd.DataFrame:
    """Every review in the category, cleaned, one row per review."""
    df = pd.read_json(path, lines=True)
    df = df[["user_id", "parent_asin", "rating", "title", "text", "timestamp"]].copy()
    df["text"] = (df["title"].fillna("") + ". " + df["text"].fillna("")).str.strip()
    df["text"] = df["text"].str.replace(r"<br\s*/?>", " ", regex=True)
    df = df[df["text"].str.len() >= min_chars]
    df = df.drop_duplicates(["user_id", "parent_asin", "text"])
    df["date"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.drop(columns=["title", "timestamp"]).sort_values("date").reset_index(drop=True)
    df["review_id"] = range(len(df))
    return df


def time_split(df: pd.DataFrame, cutoff: str):
    """History = everything before `cutoff`; targets = later reviews by users seen in history.

    Every feature, graph and community is computed from history only, so the
    model never sees anything written on or after the cutoff.
    """
    hist = df[df["date"] < cutoff]
    post = df[df["date"] >= cutoff]
    target = post[post["user_id"].isin(set(hist["user_id"]))]
    return hist, target


def load_reviews(path: str, min_reviews: int = 3, min_chars: int = 20) -> pd.DataFrame:
    """Users with at least `min_reviews` reviews (used by the descriptive analysis)."""
    df = load_raw(path, min_chars)
    counts = df["user_id"].value_counts()
    df = df[df["user_id"].isin(counts[counts >= min_reviews].index)]
    return df.sort_values(["user_id", "date"]).reset_index(drop=True)
