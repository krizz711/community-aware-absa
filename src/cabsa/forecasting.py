"""Aspect-level opinion forecasting with community-anchored partial pooling.

Task: for a user u who will review product p, predict BEFORE the review is
written (a) the star rating and (b) the sentiment u will express on each
aspect a of p. Only reviews written before the cutoff are used.

Proposed model (hierarchical baseline predictor; "CAP" = community-anchored pooling):

    s_hat(u, p, a) = mu_a + b_pa + b_ca + b_ua

  mu_a  global mean sentiment on aspect a
  b_pa  product p's offset on a (other reviewers of p), shrunk:   sum / (n + lam_p)
  b_ca  offset of u's community on a, computed from the OTHER members only:
        residuals (s - mu_a - b_pa) of members, shrunk with lam_c
  b_ua  u's own residual AFTER the community offset, shrunk with lam_u, so a
        user with little history falls back to the community, and a user with
        a long history overrides it

Baselines drop terms: product-only (mu_a + b_pa), user-only (b_ua shrunk to 0),
community-only. The same structure is used for the star rating (mu + b_p + b_c + b_u).
A gradient-boosting model on the same statistics is the learned variant.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score

from .aspects import ASPECT_LEXICON

LAM_P = 5.0  # product-level shrinkage (fixed)
_ASPECT_CODE = {a: i for i, a in enumerate(sorted(ASPECT_LEXICON))}


def _ratio(s, n, lam):
    return np.where(n > 0, s / (n + lam), 0.0)


def residualize(hist: pd.DataFrame, m_hist: pd.DataFrame):
    """Remove global and product effects so what is left is personal taste.

    Returns review frame h (column e = rating residual) and mention frame mh
    (column e = aspect-sentiment residual), plus the global/product tables.
    """
    mu = hist["rating"].mean()
    h = hist[["review_id", "user_id", "parent_asin", "rating"]].copy()
    ps = h.assign(r=h["rating"] - mu).groupby("parent_asin")["r"].agg(["sum", "count"])
    h["e"] = h["rating"] - mu - h["parent_asin"].map(ps["sum"] / (ps["count"] + LAM_P))

    mh = m_hist.merge(h[["review_id", "user_id", "parent_asin"]], on="review_id")
    mu_a = mh.groupby("aspect")["score"].mean()
    mh["d"] = mh["score"] - mh["aspect"].map(mu_a)
    pa = mh.groupby(["parent_asin", "aspect"])["d"].agg(["sum", "count"])
    mh = mh.join((pa["sum"] / (pa["count"] + LAM_P)).rename("b_pa"), on=["parent_asin", "aspect"])
    mh["e"] = mh["d"] - mh["b_pa"]
    return h, mh, {"mu": mu, "mu_a": mu_a, "ps": ps, "pa": pa}


def taste_mentions(h: pd.DataFrame, mh: pd.DataFrame) -> pd.DataFrame:
    """Profile input for 'taste' communities: aspect residuals + one 'overall' row per review."""
    overall = pd.DataFrame({"review_id": h["review_id"], "aspect": "overall",
                            "score": (h["e"] / 2).clip(-1, 1)})
    asp = mh[["review_id", "aspect"]].assign(score=mh["e"].clip(-1, 1))
    return pd.concat([asp, overall], ignore_index=True)


def _join_sums(frame, table, keys, prefix):
    t = table.rename(columns={"sum": f"{prefix}_sum", "count": f"{prefix}_n"})
    frame = frame.join(t, on=keys)
    return frame.fillna({f"{prefix}_sum": 0, f"{prefix}_n": 0})


def compute_stats(hist: pd.DataFrame, m_hist: pd.DataFrame, target: pd.DataFrame,
                  m_target: pd.DataFrame, part: dict,
                  community_rows=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sufficient statistics (sums and counts) for every target review and target mention.

    Community sums always exclude the target user's own history, so the
    community level and the user level carry separate information.
    `community_rows` (set of review ids) restricts the community-level sums to
    those reviews; pass the fold NOT used to form the communities to get
    cross-fitted community offsets.
    """
    h, mh, g = residualize(hist, m_hist)
    h["community"] = h["user_id"].map(part)
    mh["community"] = mh["user_id"].map(part)
    hc, mhc = h.dropna(subset=["community"]), mh.dropna(subset=["community"])
    if community_rows is not None:
        hc = hc[hc["review_id"].isin(community_rows)]
        mhc = mhc[mhc["review_id"].isin(community_rows)]
    csize = pd.Series(part).value_counts()
    agg = lambda d, k: d.groupby(k)["e"].agg(["sum", "count"])

    # ---- review level
    t = target[["review_id", "user_id", "parent_asin", "rating"]].copy()
    t["community"] = t["user_id"].map(part)
    t["mu"] = g["mu"]
    t = _join_sums(t, g["ps"], "parent_asin", "p")
    t = _join_sums(t, agg(h, "user_id"), "user_id", "u")
    t = _join_sums(t, agg(hc, "community"), "community", "c")
    t = _join_sums(t, agg(hc, ["community", "parent_asin"]), ["community", "parent_asin"], "cp")
    t = _join_sums(t, agg(hc, "user_id"), "user_id", "uc")
    t = _join_sums(t, agg(hc, ["user_id", "parent_asin"]), ["user_id", "parent_asin"], "upc")
    in_c = t["community"].notna()
    for lvl, own in [("c", "uc"), ("cp", "upc")]:        # other members only
        t[f"{lvl}_sum"] -= np.where(in_c, t[f"{own}_sum"], 0)
        t[f"{lvl}_n"] -= np.where(in_c, t[f"{own}_n"], 0)
    t["c_size"] = t["community"].map(csize).fillna(0)

    # ---- aspect level
    mt = m_target.merge(t[["review_id", "user_id", "parent_asin", "community", "u_n"]],
                        on="review_id")
    mt["mu_a"] = mt["aspect"].map(g["mu_a"])
    mt = _join_sums(mt, g["pa"], ["parent_asin", "aspect"], "pa")
    mt = _join_sums(mt, agg(mh, ["user_id", "aspect"]), ["user_id", "aspect"], "ua")
    mt = _join_sums(mt, agg(mhc, ["community", "aspect"]), ["community", "aspect"], "ca")
    mt = _join_sums(mt, agg(mhc, ["community", "parent_asin", "aspect"]),
                    ["community", "parent_asin", "aspect"], "cpa")
    mt = _join_sums(mt, agg(mhc, ["user_id", "aspect"]), ["user_id", "aspect"], "uac")
    mt = _join_sums(mt, agg(mhc, ["user_id", "parent_asin", "aspect"]),
                    ["user_id", "parent_asin", "aspect"], "upac")
    in_c = mt["community"].notna()
    for lvl, own in [("ca", "uac"), ("cpa", "upac")]:
        mt[f"{lvl}_sum"] -= np.where(in_c, mt[f"{own}_sum"], 0)
        mt[f"{lvl}_n"] -= np.where(in_c, mt[f"{own}_n"], 0)
    mt["c_size"] = mt["community"].map(csize).fillna(0)
    return t, mt


DEFAULT_LAM = {"c": 30.0, "cp": 3.0, "u": 10.0}


def _levels(f, pre, lam):
    """Offsets of every level for one frame; pre = '' (rating) or 'a' (aspect)."""
    b_c = _ratio(f[f"c{pre}_sum"], f[f"c{pre}_n"], lam["c"])
    b_cp = _ratio(f[f"cp{pre}_sum"] - f[f"cp{pre}_n"] * b_c, f[f"cp{pre}_n"], lam["cp"])
    b_u = _ratio(f[f"u{pre}_sum"], f[f"u{pre}_n"], lam["u"])
    b_uh = _ratio(f[f"u{pre}_sum"] - f[f"u{pre}_n"] * b_c, f[f"u{pre}_n"], lam["u"])
    return b_c, b_cp, b_u, b_uh


def predict(t: pd.DataFrame, mt: pd.DataFrame, lam: dict | None = None) -> dict:
    """Closed-form predictions of every variant, for reviews (rating) and mentions (score)."""
    lam = {**DEFAULT_LAM, **(lam or {})}
    out = {}
    for level, f, base in [("rating", t, t["mu"] + _ratio(t["p_sum"], t["p_n"], LAM_P)),
                           ("aspect", mt, mt["mu_a"] + _ratio(mt["pa_sum"], mt["pa_n"], LAM_P))]:
        b_c, b_cp, b_u, b_uh = _levels(f, "" if level == "rating" else "a", lam)
        out[level] = {"product": base,
                      "+user": base + b_u,
                      "+community": base + b_c,
                      "+community x product": base + b_c + b_cp,
                      "CAP (full)": base + b_c + b_cp + b_uh}
    lo_hi = {"rating": (1, 5), "aspect": (-1, 1)}
    return {k: {n: np.clip(np.asarray(v, float), *lo_hi[k]) for n, v in d.items()}
            for k, d in out.items()}


def tune_lambdas(t, mt, grid=(1, 3, 10, 30, 100), target: str = "aspect"):
    """Grid-search (lam_c, lam_cp, lam_u) for CAP on the validation period."""
    rows = []
    for lc, lcp, lu in itertools.product(grid, grid, grid):
        lam = {"c": lc, "cp": lcp, "u": lu}
        p = predict(t, mt, lam)
        rows.append({**lam, "rmse_aspect": rmse(mt["score"], p["aspect"]["CAP (full)"]),
                     "rmse_rating": rmse(t["rating"], p["rating"]["CAP (full)"])})
    g = pd.DataFrame(rows)
    best = g.loc[g[f"rmse_{target}"].idxmin()]
    return {k: float(best[k]) for k in ("c", "cp", "u")}, g


def rmse(y, p) -> float:
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def auc_pos_neg(y_score, pred, level: str) -> float:
    """AUC for positive vs negative (neutral dropped): aspect score +/-0.05, rating >=4 vs <=2."""
    y = np.asarray(y_score, float)
    if level == "rating":
        keep, lab = (y != 3), (y >= 4)
    else:
        keep, lab = (np.abs(y) > 0.05), (y > 0.05)
    return float(roc_auc_score(lab[keep], np.asarray(pred)[keep]))


def history_bucket(n) -> pd.Series:
    return pd.cut(pd.Series(n), [0, 1, 3, np.inf], labels=["1 review", "2-3 reviews", "4+ reviews"])


def evaluate(t, mt, preds, strata: bool = True) -> pd.DataFrame:
    rows = []
    for level, frame, y in [("rating", t, "rating"), ("aspect", mt, "score")]:
        b = history_bucket(frame["u_n"].to_numpy())
        groups = [("all", np.ones(len(frame), bool))]
        if strata:
            groups += [(str(c), (b == c).to_numpy()) for c in b.cat.categories]
        for gname, mask in groups:
            for name, p in preds[level].items():
                rows.append({"level": level, "users": gname, "model": name, "n": int(mask.sum()),
                             "rmse": rmse(frame[y][mask], p[mask]),
                             "auc": auc_pos_neg(frame[y][mask], p[mask], level)})
    return pd.DataFrame(rows)


def cluster_bootstrap(frame, y, p_a, p_b, metric: str = "rmse", level: str = "aspect",
                      n_boot: int = 500, seed: int = 0) -> np.ndarray:
    """Paired bootstrap over USERS of metric(p_a) - metric(p_b); returns [2.5, 50, 97.5] pct."""
    rng = np.random.default_rng(seed)
    groups = frame.groupby("user_id").indices
    idx_list = list(groups.values())
    y, p_a, p_b = np.asarray(y, float), np.asarray(p_a, float), np.asarray(p_b, float)
    out = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(idx_list), len(idx_list))
        ix = np.concatenate([idx_list[i] for i in pick])
        if metric == "rmse":
            out.append(rmse(y[ix], p_a[ix]) - rmse(y[ix], p_b[ix]))
        else:
            out.append(auc_pos_neg(y[ix], p_a[ix], level) - auc_pos_neg(y[ix], p_b[ix], level))
    return np.percentile(out, [2.5, 50, 97.5])


# ---------------------------------------------------------------- learned variant

def gbm_features(f: pd.DataFrame, level: str, which: str) -> pd.DataFrame:
    """Raw history statistics; the model learns how much to trust each level."""
    safe = lambda s, n: np.where(n > 0, s / np.maximum(n, 1), 0.0)
    a = "a" if level == "aspect" else ""
    if level == "aspect":
        X = pd.DataFrame({"mu_a": f["mu_a"], "p_mean": safe(f["pa_sum"], f["pa_n"]), "p_n": f["pa_n"],
                          "aspect": f["aspect"].map(_ASPECT_CODE).fillna(-1).astype(int)})
    else:
        X = pd.DataFrame({"p_mean": safe(f["p_sum"], f["p_n"]), "p_n": f["p_n"]})
    if "user" in which:
        X["u_mean"], X["u_n"] = safe(f[f"u{a}_sum"], f[f"u{a}_n"]), f[f"u{a}_n"]
        X["u_reviews"] = f["u_n"]
    if "community" in which:
        X["c_mean"], X["c_n"] = safe(f[f"c{a}_sum"], f[f"c{a}_n"]), f[f"c{a}_n"]
        X["cp_mean"], X["cp_n"] = safe(f[f"cp{a}_sum"], f[f"cp{a}_n"]), f[f"cp{a}_n"]
        X["c_size"] = f["c_size"]
    return X


def fit_gbm(f_train, f_test, level: str, which: str, seed: int = 0):
    """Train on the validation period, predict the test period."""
    y = "score" if level == "aspect" else "rating"
    Xtr, Xte = gbm_features(f_train, level, which), gbm_features(f_test, level, which)
    cat = ["aspect"] if level == "aspect" else None
    gbm = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
                                        min_samples_leaf=50, l2_regularization=1.0,
                                        categorical_features=cat, early_stopping=True,
                                        validation_fraction=0.15, random_state=seed)
    gbm.fit(Xtr, f_train[y])
    lo, hi = (-1, 1) if level == "aspect" else (1, 5)
    return np.clip(gbm.predict(Xte), lo, hi), gbm


GBM_SETS = {"GBM product": "", "GBM +user": "user", "GBM +community": "community",
            "GBM +user+community": "user community"}
