"""Secondary task: which aspects will a user's next review talk about?

Same pooling idea as forecasting.py, applied to aspect *attention* (a binary
review x aspect matrix) instead of aspect *sentiment*.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .aspects import ASPECT_LEXICON

ASPECTS = sorted(ASPECT_LEXICON)


def review_aspect_matrix(reviews: pd.DataFrame, mentions: pd.DataFrame) -> np.ndarray:
    m = mentions[mentions["review_id"].isin(reviews["review_id"])]
    Y = pd.crosstab(m["review_id"], m["aspect"]).reindex(index=reviews["review_id"], columns=ASPECTS)
    return (Y.fillna(0).to_numpy() > 0).astype(float)


def attention_stats(hist, m_hist, target, m_target, part, lam_p: float = 5.0):
    Yh, Yt = review_aspect_matrix(hist, m_hist), review_aspect_matrix(target, m_target)
    g = Yh.mean(0)
    H = pd.DataFrame(Yh, columns=ASPECTS)
    H["user_id"], H["p"] = hist["user_id"].to_numpy(), hist["parent_asin"].to_numpy()
    H["c"] = hist["user_id"].map(part).to_numpy()
    look = lambda tab, key: tab.reindex(key).fillna(0).to_numpy()
    t = target[["user_id", "parent_asin"]].reset_index(drop=True)
    c = t["user_id"].map(part)
    P, Pn = H.groupby("p")[ASPECTS].sum(), H.groupby("p").size()
    U, Un = H.groupby("user_id")[ASPECTS].sum(), H.groupby("user_id").size()
    Hc = H.dropna(subset=["c"])
    C, Cn = Hc.groupby("c")[ASPECTS].sum(), Hc.groupby("c").size()
    us, uk = look(U, t["user_id"]), look(Un, t["user_id"])[:, None]
    has = c.notna().to_numpy()[:, None]
    cs = look(C, c) - np.where(has, us, 0)
    ck = look(Cn, c)[:, None] - np.where(has, uk, 0)
    base = (look(P, t["parent_asin"]) + lam_p * g) / (look(Pn, t["parent_asin"])[:, None] + lam_p)
    return {"Y": Yt, "g": g, "base": base, "us": us, "uk": uk, "cs": cs, "ck": ck,
            "u_n": uk[:, 0]}


def attention_predict(s, lam_c: float, lam_u: float) -> dict:
    g = s["g"]
    comm = (s["cs"] + lam_c * g) / (s["ck"] + lam_c)
    user = (s["us"] + lam_u * g) / (s["uk"] + lam_u)
    user_h = (s["us"] + lam_u * comm) / (s["uk"] + lam_u)
    lift = lambda x: np.clip(s["base"] * x / g, 1e-4, 1 - 1e-4)
    return {"product": np.clip(s["base"], 1e-4, 1 - 1e-4), "+user": lift(user),
            "+community": lift(comm), "CAP (community+user)": lift(user_h)}


def attention_scores(Y, pred, mask=None) -> tuple[float, float]:
    """(mean per-aspect AUC, mean log-loss)."""
    if mask is not None:
        Y, pred = Y[mask], pred[mask]
    aucs = [roc_auc_score(Y[:, j], pred[:, j]) for j in range(Y.shape[1]) if 0 < Y[:, j].sum() < len(Y)]
    ll = -np.mean(Y * np.log(pred) + (1 - Y) * np.log(1 - pred))
    return float(np.mean(aucs)), float(ll)
