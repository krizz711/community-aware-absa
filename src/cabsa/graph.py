"""Weighted user-aspect graph, user-user similarity graph, Louvain communities.

1. Bipartite user-aspect graph: edge (u, a) with weight = number of times u
   mentioned aspect a, and attribute `opinion` = u's mean sentiment on a.
2. User-user projection: two users are similar when they talk about the same
   aspects (interest overlap) AND feel the same way about them (opinion
   agreement):
       w(u, v) = sum_a min(p_ua, p_va) * (1 - |s_ua - s_va| / 2)
   p_ua = share of u's mentions about aspect a, s_ua in [-1, 1] = u's mean
   sentiment on a. With agreement=False the second factor is dropped
   (interest-only ablation).
3. Each core user keeps its k strongest neighbours; Louvain maximises weighted
   modularity on that sparse graph.
4. Users outside the core (e.g. one-review users) are assigned inductively:
   the community with the largest total similarity among their k nearest core
   users.
"""
from __future__ import annotations

from collections import defaultdict

import networkx as nx
import numpy as np
import pandas as pd


def user_aspect_profiles(mentions: pd.DataFrame, reviews: pd.DataFrame):
    """Return (users, aspects, P = interest shares, S = mean opinion, N = counts)."""
    m = mentions.merge(reviews[["review_id", "user_id"]], on="review_id")
    g = m.groupby(["user_id", "aspect"])["score"].agg(["count", "mean"]).reset_index()
    N = g.pivot(index="user_id", columns="aspect", values="count").fillna(0)
    S = g.pivot(index="user_id", columns="aspect", values="mean").reindex_like(N).fillna(0)
    P = N.div(N.sum(axis=1), axis=0)
    return N.index.to_numpy(), N.columns.to_numpy(), P.to_numpy(), S.to_numpy(), N.to_numpy()


def bipartite_graph(users, aspects, N, S) -> nx.Graph:
    B = nx.Graph()
    B.add_nodes_from((f"u:{u}" for u in users), bipartite=0)
    B.add_nodes_from((f"a:{a}" for a in aspects), bipartite=1)
    rows, cols = np.nonzero(N)
    for i, j in zip(rows, cols):
        B.add_edge(f"u:{users[i]}", f"a:{aspects[j]}", weight=float(N[i, j]),
                   opinion=float(S[i, j]))
    return B


def knn(Pq, Sq, Pc, Sc, k: int = 15, agreement: bool = True, same_set: bool = False,
        chunk: int = 1024):
    """k most similar core users for every query user (GPU via torch when available)."""
    try:
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        T = lambda x: torch.as_tensor(np.asarray(x, dtype=np.float32), device=dev)
        Pc_, Sc_ = T(Pc), T(Sc)
        idx_out, w_out = [], []
        for s in range(0, len(Pq), chunk):
            p, q = T(Pq[s:s + chunk]), T(Sq[s:s + chunk])
            W = torch.zeros((len(p), len(Pc)), device=dev)
            for a in range(Pc_.shape[1]):
                ov = torch.minimum(p[:, a, None], Pc_[None, :, a])
                if agreement:
                    ov = ov * (1 - (q[:, a, None] - Sc_[None, :, a]).abs() / 2)
                W += ov
            if same_set:
                r = torch.arange(len(p), device=dev)
                W[r, r + s] = -1
            w, i = torch.topk(W, k, dim=1)
            idx_out.append(i.cpu().numpy()); w_out.append(w.cpu().numpy())
        return np.vstack(idx_out), np.vstack(w_out)
    except ImportError:
        idx_out, w_out = [], []
        Pc, Sc = np.asarray(Pc, np.float32), np.asarray(Sc, np.float32)
        for s in range(0, len(Pq), chunk // 4):
            p = np.asarray(Pq[s:s + chunk // 4], np.float32)
            q = np.asarray(Sq[s:s + chunk // 4], np.float32)
            W = np.zeros((len(p), len(Pc)), np.float32)
            for a in range(Pc.shape[1]):
                ov = np.minimum(p[:, a, None], Pc[None, :, a])
                if agreement:
                    ov *= 1 - np.abs(q[:, a, None] - Sc[None, :, a]) / 2
                W += ov
            if same_set:
                W[np.arange(len(p)), np.arange(len(p)) + s] = -1
            top = np.argpartition(-W, k, axis=1)[:, :k]
            idx_out.append(top); w_out.append(np.take_along_axis(W, top, axis=1))
        return np.vstack(idx_out), np.vstack(w_out)


def user_graph(users, nbr_idx, nbr_w, min_weight: float = 0.05) -> nx.Graph:
    G = nx.Graph()
    G.add_nodes_from(users)
    for i in range(len(users)):
        for j, w in zip(nbr_idx[i], nbr_w[i]):
            if w < min_weight or i == j:
                continue
            u, v = users[i], users[j]
            if G.has_edge(u, v):
                G[u][v]["weight"] = max(G[u][v]["weight"], float(w))
            else:
                G.add_edge(u, v, weight=float(w))
    G.remove_nodes_from(list(nx.isolates(G)))
    return G


def louvain(G: nx.Graph, resolution: float = 1.0, seed: int = 0) -> dict:
    comms = nx.community.louvain_communities(G, weight="weight", resolution=resolution, seed=seed)
    comms = sorted(comms, key=len, reverse=True)
    return {u: cid for cid, c in enumerate(comms) for u in c}


def modularity(G: nx.Graph, part: dict) -> float:
    groups: dict[int, set] = defaultdict(set)
    for u, c in part.items():
        if u in G:
            groups[c].add(u)
    return nx.community.modularity(G, groups.values(), weight="weight")


def assign_inductive(query_users, nbr_idx, nbr_w, core_users, core_part: dict) -> dict:
    """Community = argmax over communities of summed similarity to the k nearest core users."""
    out = {}
    for u, idx, w in zip(query_users, nbr_idx, nbr_w):
        votes: dict[int, float] = defaultdict(float)
        for j, wj in zip(idx, w):
            c = core_part.get(core_users[j])
            if c is not None and wj > 0:
                votes[c] += wj
        if votes:
            out[u] = max(votes, key=votes.get)
    return out


def kmeans_partition(users, P, S, n_clusters: int, seed: int = 0) -> dict:
    """Ablation: cluster the same profiles with k-means instead of a graph."""
    from sklearn.cluster import KMeans

    X = np.hstack([P, P * S])
    lab = KMeans(n_clusters=n_clusters, n_init=4, random_state=seed).fit_predict(X)
    return dict(zip(users, lab.tolist()))


def random_partition(part: dict, seed: int = 0) -> dict:
    """Same community sizes, members shuffled: the control for 'is it the structure?'."""
    rng = np.random.default_rng(seed)
    users = list(part)
    labels = np.array([part[u] for u in users])
    rng.shuffle(labels)
    return dict(zip(users, labels.tolist()))


def build_communities(hist: pd.DataFrame, mentions: pd.DataFrame, need_users,
                      k: int = 15, resolution: float = 1.0, agreement: bool = True,
                      min_core_reviews: int = 2, seed: int = 0):
    """Communities over `hist`, plus an inductive assignment for every user in `need_users`.

    Returns (part, info): part maps user -> community for core users and every
    needed user that has at least one aspect mention in history.
    """
    users, aspects, P, S, N = user_aspect_profiles(mentions, hist)
    n_rev = hist["user_id"].value_counts()
    core_mask = n_rev.reindex(users).fillna(0).to_numpy() >= min_core_reviews
    cu, Pc, Sc = users[core_mask], P[core_mask], S[core_mask]
    idx, w = knn(Pc, Sc, Pc, Sc, k=k, agreement=agreement, same_set=True)
    G = user_graph(cu, idx, w)
    core_part = louvain(G, resolution=resolution, seed=seed)

    need = np.array([u for u in set(need_users) if u not in core_part])
    pos = pd.Series(np.arange(len(users)), index=users)
    need = need[pd.Index(need).isin(pos.index)] if len(need) else need
    part = dict(core_part)
    if len(need):
        qi = pos[need].to_numpy()
        qidx, qw = knn(P[qi], S[qi], Pc, Sc, k=k, agreement=agreement)
        part.update(assign_inductive(need, qidx, qw, cu, core_part))
    info = {"graph": G, "core_part": core_part, "users": users, "aspects": aspects,
            "P": P, "S": S, "N": N, "core_mask": core_mask}
    return part, info
