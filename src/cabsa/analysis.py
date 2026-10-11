"""How does sentiment inside a community differ from overall sentiment?"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import jensenshannon


def _bh(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values."""
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    ranked = p[order] * len(p) / (np.arange(len(p)) + 1)
    adj = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(adj)
    out[order] = np.minimum(adj, 1.0)
    return out


def tag_mentions(mentions: pd.DataFrame, reviews: pd.DataFrame, part: dict) -> pd.DataFrame:
    m = mentions.merge(reviews[["review_id", "user_id"]], on="review_id")
    m["community"] = m["user_id"].map(part)
    return m.dropna(subset=["community"]).astype({"community": int})


def community_summary(m: pd.DataFrame, part: dict, min_users: int = 30) -> pd.DataFrame:
    """Size, interest divergence from the population, and signature aspects."""
    sizes = pd.Series(part).value_counts()
    keep = sizes[sizes >= min_users].index
    glob_share = m["aspect"].value_counts(normalize=True)
    rows = []
    for c in keep:
        mc = m[m["community"] == c]
        share = mc["aspect"].value_counts(normalize=True).reindex(glob_share.index, fill_value=0)
        lift = (share / glob_share).sort_values(ascending=False)
        rows.append({"community": c, "users": int(sizes[c]), "mentions": len(mc),
                     "mean_sentiment": mc["score"].mean(),
                     "js_interest": jensenshannon(share, glob_share, base=2),
                     "signature": ", ".join(lift.index[:3])})
    return pd.DataFrame(rows).sort_values("users", ascending=False).reset_index(drop=True)


def aspect_sentiment_shift(m: pd.DataFrame, communities, min_mentions: int = 30) -> pd.DataFrame:
    """Community vs rest-of-population share of positive mentions, per aspect.

    Two-proportion z-test for every (community, aspect) cell with enough
    mentions, Benjamini-Hochberg corrected across all cells.
    """
    rows = []
    for c in communities:
        inside = m["community"] == c
        for a, ma in m.groupby("aspect"):
            ins, out = ma[inside[ma.index]], ma[~inside[ma.index]]
            if len(ins) < min_mentions or len(out) < min_mentions:
                continue
            p1, p2 = (ins["polarity"] == 1).mean(), (out["polarity"] == 1).mean()
            n1, n2 = len(ins), len(out)
            pp = (p1 * n1 + p2 * n2) / (n1 + n2)
            se = np.sqrt(pp * (1 - pp) * (1 / n1 + 1 / n2))
            z = (p1 - p2) / se if se > 0 else 0.0
            rows.append({"community": c, "aspect": a, "n_comm": n1, "pos_comm": p1,
                         "pos_rest": p2, "diff": p1 - p2,
                         "p": 2 * stats.norm.sf(abs(z))})
    df = pd.DataFrame(rows)
    if len(df):
        df["p_bh"] = _bh(df["p"].to_numpy())
        df["significant"] = df["p_bh"] < 0.05
    return df


def heatmap_table(m: pd.DataFrame, communities) -> pd.DataFrame:
    """Mean aspect sentiment per community minus the global mean (rows = community)."""
    glob = m.groupby("aspect")["score"].mean()
    t = (m[m["community"].isin(communities)]
         .groupby(["community", "aspect"])["score"].mean().unstack())
    return t.sub(glob, axis=1).reindex(communities)
