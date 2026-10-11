"""Does community information improve sentiment prediction?

Task: predict whether a user's held-out (most recent) review is positive
(4-5 stars) or negative (1-2 stars). All models share the same TF-IDF text
features; they differ only in the prior features appended to the text:

  text            text only
  +user           the user's own history (mean rating, opinion on the aspects
                  this review mentions) - the personalisation baseline
  +community      what OTHER members of the user's community said
                  (community mean rating, community opinion on the aspects
                  this review mentions); the user's own reviews are excluded
  +random         same as +community but on a size-matched random partition;
                  if this helps as much, the gain is not from the structure
  +user+community both priors

Train rows get leave-one-out priors so they are computed the same way as for
test rows; no test rating is used anywhere.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler


def _review_aspects(mentions: pd.DataFrame) -> pd.DataFrame:
    return mentions.groupby(["review_id", "aspect"])["score"].agg(["sum", "count"]).reset_index()


def user_prior_features(df: pd.DataFrame, ra: pd.DataFrame) -> pd.DataFrame:
    tr = df[df["split"] == "train"]
    gmean = tr["rating"].mean()
    us = tr.groupby("user_id")["rating"].agg(["sum", "count"])
    f = df[["review_id", "user_id", "rating", "split"]].join(us, on="user_id")
    is_tr = f["split"] == "train"
    num = f["sum"] - np.where(is_tr, f["rating"], 0)
    den = f["count"] - is_tr.astype(int)
    f["user_rating"] = np.where(den > 0, num / den.replace(0, 1), gmean)

    # user's opinion on each aspect, minus this review's own contribution
    ra_u = ra.merge(df[["review_id", "user_id", "split"]], on="review_id")
    ua = ra_u[ra_u["split"] == "train"].groupby(["user_id", "aspect"])[["sum", "count"]].sum()
    x = ra_u.join(ua, on=["user_id", "aspect"], rsuffix="_u")
    own = x["split"] == "train"
    num = x["sum_u"] - np.where(own, x["sum"], 0)
    den = x["count_u"] - np.where(own, x["count"], 0)
    x["user_aspect"] = np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)
    f = f.merge(x.groupby("review_id")["user_aspect"].mean(), on="review_id", how="left")
    f["user_aspect"] = f["user_aspect"].fillna(0.0)
    return f.set_index("review_id")[["user_rating", "user_aspect"]]


def community_prior_features(df: pd.DataFrame, ra: pd.DataFrame, part: dict) -> pd.DataFrame:
    tr = df[df["split"] == "train"].copy()
    gmean = tr["rating"].mean()
    comm = df["user_id"].map(part)
    tr["community"] = tr["user_id"].map(part)
    cs = tr.dropna(subset=["community"]).groupby("community")["rating"].agg(["sum", "count"])
    us = tr.groupby("user_id")["rating"].agg(["sum", "count"])

    f = pd.DataFrame({"review_id": df["review_id"], "user_id": df["user_id"], "community": comm})
    f = f.join(cs, on="community").join(us, on="user_id", rsuffix="_u")
    num = f["sum"] - f["sum_u"].fillna(0)
    den = f["count"] - f["count_u"].fillna(0)
    f["comm_rating"] = np.where(den > 0, num / np.where(den > 0, den, 1), gmean)

    ra_c = ra.merge(df[["review_id", "user_id", "split"]], on="review_id")
    ra_c["community"] = ra_c["user_id"].map(part)
    trc = ra_c[ra_c["split"] == "train"]
    ca = trc.groupby(["community", "aspect"])[["sum", "count"]].sum()
    ua = trc.groupby(["user_id", "aspect"])[["sum", "count"]].sum()
    x = ra_c.join(ca, on=["community", "aspect"], rsuffix="_c").join(ua, on=["user_id", "aspect"], rsuffix="_u")
    num = x["sum_c"] - x["sum_u"].fillna(0)
    den = x["count_c"] - x["count_u"].fillna(0)
    x["comm_aspect"] = np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)
    f = f.merge(x.groupby("review_id")["comm_aspect"].mean(), on="review_id", how="left")
    f["comm_aspect"] = f["comm_aspect"].fillna(0.0)
    f["comm_size"] = np.log1p(f["community"].map(pd.Series(part).value_counts()).fillna(0))
    return f.set_index("review_id")[["comm_rating", "comm_aspect", "comm_size"]]


def _macro_f1(y: np.ndarray, yhat: np.ndarray) -> float:
    f = []
    for c in (0, 1):
        tp = np.sum((yhat == c) & (y == c))
        prec_den, rec_den = np.sum(yhat == c), np.sum(y == c)
        p = tp / prec_den if prec_den else 0.0
        r = tp / rec_den if rec_den else 0.0
        f.append(2 * p * r / (p + r) if p + r else 0.0)
    return float(np.mean(f))


def run(df: pd.DataFrame, mentions: pd.DataFrame, part: dict, rand_part: dict,
        n_boot: int = 1000, seed: int = 0, short_words: int = 12) -> tuple[pd.DataFrame, dict]:
    df = df.copy()
    df["y"] = df["rating"].map({1: 0, 2: 0, 4: 1, 5: 1})
    ra = _review_aspects(mentions)
    U = user_prior_features(df, ra)
    C = community_prior_features(df, ra, part)
    R = community_prior_features(df, ra, rand_part).add_prefix("rand_")

    d = df.dropna(subset=["y"]).set_index("review_id")
    d = d.join(U).join(C).join(R)
    tr, te = d[d["split"] == "train"], d[d["split"] == "test"]

    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=50000, sublinear_tf=True)
    Xtr_t, Xte_t = vec.fit_transform(tr["text"]), vec.transform(te["text"])

    sets = {
        "text": [],
        "+user": ["user_rating", "user_aspect"],
        "+community": ["comm_rating", "comm_aspect", "comm_size"],
        "+random": ["rand_comm_rating", "rand_comm_aspect", "rand_comm_size"],
        "+user+community": ["user_rating", "user_aspect", "comm_rating", "comm_aspect", "comm_size"],
    }
    # short reviews (title + a few words) are where the text alone is weakest
    short = (te["text"].str.split().str.len() <= short_words).to_numpy()
    probs, rows = {}, []
    for name, cols in sets.items():
        if cols:
            sc = StandardScaler().fit(tr[cols])
            Xtr = sparse.hstack([Xtr_t, sparse.csr_matrix(sc.transform(tr[cols]))]).tocsr()
            Xte = sparse.hstack([Xte_t, sparse.csr_matrix(sc.transform(te[cols]))]).tocsr()
        else:
            Xtr, Xte = Xtr_t, Xte_t
        clf = LogisticRegression(C=4.0, max_iter=2000, class_weight="balanced")
        clf.fit(Xtr, tr["y"])
        p = clf.predict_proba(Xte)[:, 1]
        probs[name] = p
        rows.append({"model": name, "macro_f1": f1_score(te["y"], p > 0.5, average="macro"),
                     "auc": roc_auc_score(te["y"], p),
                     "f1_short": f1_score(te["y"][short], p[short] > 0.5, average="macro"),
                     "auc_short": roc_auc_score(te["y"][short], p[short]), "n_test": len(te)})
    res = pd.DataFrame(rows)

    # paired bootstrap of the gain over text-only
    rng = np.random.default_rng(seed)
    y = te["y"].to_numpy()
    idx = rng.integers(0, len(y), size=(n_boot, len(y)))
    ci = {}
    for name in sets:
        if name == "text":
            continue
        gains = []
        for b in idx:
            if y[b].min() == y[b].max():
                continue
            gains.append(_macro_f1(y[b], (probs[name][b] > 0.5).astype(int))
                         - _macro_f1(y[b], (probs["text"][b] > 0.5).astype(int)))
        ci[name] = np.percentile(gains, [2.5, 50, 97.5])
    res["f1_gain_ci95"] = res["model"].map(
        lambda n: "" if n == "text" else f"{ci[n][1]:+.4f} [{ci[n][0]:+.4f}, {ci[n][2]:+.4f}]")

    # priors only: how much does each source know about the next review
    # before reading a single word of it?
    prior_rows = []
    for name, cols in [("user only", sets["+user"][:1]),
                       ("community only", sets["+community"][:1]),
                       ("random only", sets["+random"][:1]),
                       ("user+community", ["user_rating", "comm_rating"])]:
        clf = LogisticRegression(max_iter=1000, class_weight="balanced").fit(tr[cols], tr["y"])
        p = clf.predict_proba(te[cols])[:, 1]
        prior_rows.append({"model": name, "auc": roc_auc_score(te["y"], p),
                           "macro_f1": f1_score(te["y"], p > 0.5, average="macro")})
    extras = {"probs": probs, "y": y, "test": te, "short": short,
              "priors_only": pd.DataFrame(prior_rows)}
    return res, extras
