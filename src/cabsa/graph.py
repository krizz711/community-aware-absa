"""Weighted user-aspect graph, user-user similarity graph, Louvain communities.

1. Bipartite user-aspect graph: edge (u, a) with weight = number of times u
   mentioned aspect a, and attribute `opinion` = u's mean sentiment on a.
2. User-user projection: two users are similar when they talk about the same
   aspects (interest overlap) AND feel the same way about them (opinion
   agreement):
       w(u, v) = sum_a min(p_ua, p_va) * (1 - |s_ua - s_va| / 2)
   where p_ua is the share of u's mentions that are about a and s_ua is in
   [-1, 1]. Each user keeps only the k strongest neighbours, which keeps the
   graph sparse enough for Louvain.
3. Louvain (networkx) maximises weighted modularity on the projection.
"""
from __future__ import annotations

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


def similarity_knn(P: np.ndarray, S: np.ndarray, k: int = 15, chunk: int = 512):
    """Top-k neighbours of every user under the interest x agreement similarity."""
    n, d = P.shape
    nbr_idx = np.zeros((n, k), dtype=np.int64)
    nbr_w = np.zeros((n, k), dtype=np.float32)
    Pf, Sf = P.astype(np.float32), S.astype(np.float32)
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        W = np.zeros((stop - start, n), dtype=np.float32)
        for a in range(d):
            overlap = np.minimum(Pf[start:stop, a, None], Pf[None, :, a])
            agree = 1.0 - np.abs(Sf[start:stop, a, None] - Sf[None, :, a]) / 2.0
            W += overlap * agree
        W[np.arange(stop - start), np.arange(start, stop)] = -1  # no self loops
        top = np.argpartition(-W, k, axis=1)[:, :k]
        nbr_idx[start:stop] = top
        nbr_w[start:stop] = np.take_along_axis(W, top, axis=1)
    return nbr_idx, nbr_w


def user_graph(users, nbr_idx, nbr_w, min_weight: float = 0.05) -> nx.Graph:
    G = nx.Graph()
    G.add_nodes_from(users)
    for i in range(len(users)):
        for j, w in zip(nbr_idx[i], nbr_w[i]):
            if w < min_weight:
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
    groups: dict[int, set] = {}
    for u, c in part.items():
        groups.setdefault(c, set()).add(u)
    return nx.community.modularity(G, groups.values(), weight="weight")


def random_partition(part: dict, seed: int = 0) -> dict:
    """Same community sizes, members shuffled: the control for 'is it the structure?'."""
    rng = np.random.default_rng(seed)
    users = list(part)
    labels = np.array([part[u] for u in users])
    rng.shuffle(labels)
    return dict(zip(users, labels.tolist()))
