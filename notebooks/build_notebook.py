"""Generates community_aware_absa.ipynb from the cells below (keeps the notebook diff-able)."""
import nbformat as nbf

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md("""
# Community-Aware Aspect-Based Sentiment Analysis

**Team:** Munal Singh (24MID0095) · Devadharsh S (24MID0210)

**Question.** Reviewers might form *communities*: groups that care about the same product aspects and
feel the same way about them. We ask two things:

1. Does community-level sentiment differ from overall sentiment?
2. Does knowing a reviewer's community improve sentiment prediction?

**Pipeline.**
1. Amazon reviews
2. Clause-level aspect extraction
3. Aspect sentiment
4. Weighted user–aspect graph
5. User–user similarity projection
6. Louvain communities, with inductive assignment of new users
7. (a) community vs overall sentiment, tested both in-sample and cross-fitted;
   (b) forecasting a user's rating, aspect sentiment and aspect attention *before the review is written*

**Evaluation protocol.**
- Global time cutoff: every graph, community and statistic is built from reviews written before
  1 Jan 2021, and models are tested on reviews written after it.
- Hyper-parameters are tuned on a separate 2020 validation period.
- Every community result is compared with a size-matched random partition.

**Runtime.** About 15 minutes on Colab. A GPU runtime makes the graph steps faster and enables the
transformer check in section 3.
""")

md("## 0. Setup")
code("""
import os, sys
REPO = "https://github.com/krizz711/community-aware-absa.git"
try:
    import google.colab  # noqa: F401
    IN_COLAB = True
except ImportError:
    IN_COLAB = False
if IN_COLAB:
    if not os.path.exists("community-aware-absa"):
        !git clone -q {REPO}
    %cd community-aware-absa
    !pip -q install vaderSentiment
elif os.path.basename(os.getcwd()) == "notebooks":
    os.chdir("..")
sys.path.insert(0, "src")
print("working dir:", os.getcwd())
""")
code("""
import time, warnings
import numpy as np, pandas as pd, matplotlib.pyplot as plt, networkx as nx
from scipy import stats
from cabsa import data, aspects, sentiment, graph, analysis, forecasting as F, attention as A, viz
from cabsa.viz import BLUE, ORANGE, AQUA, GREY, MUTED
warnings.filterwarnings("ignore", category=FutureWarning)
viz.setup()

CATEGORY   = "All_Beauty"    # Amazon Reviews 2023 category
CUTOFF_VAL = "2020-01-01"    # validation period = 2020 (used only for tuning)
CUTOFF     = "2021-01-01"    # test period = 2021 onwards
K          = 15              # neighbours kept per user in the similarity graph
SEED       = 0
T0 = time.time()
""")

# ------------------------------------------------------------------ 1 data
md("""
## 1. Data
Amazon Reviews 2023 (Hou et al., 2024), category *All_Beauty*: every review, with user id, product id,
star rating, text and timestamp.
""")
code("""
path = data.download(CATEGORY, "data")
df = data.load_raw(path)
hist, target = data.time_split(df, CUTOFF)
print(f"{len(df):,} reviews | {df.user_id.nunique():,} users | {df.parent_asin.nunique():,} products | "
      f"{df.date.min():%Y-%m} to {df.date.max():%Y-%m}")
print(f"history (< {CUTOFF}): {len(hist):,} reviews")
print(f"test targets: {len(target):,} later reviews by {target.user_id.nunique():,} users already seen in history")
print(f"test reviews whose product already had reviews before the cutoff: "
      f"{target.parent_asin.isin(set(hist.parent_asin)).mean():.1%}")
""")
code("""
fig, ax = plt.subplots(1, 3, figsize=(13, 3.3))
df.set_index("date").resample("QE").size().plot(ax=ax[0], color=BLUE, lw=2)
for c, lab in [(CUTOFF_VAL, "validation"), (CUTOFF, "test")]:
    ax[0].axvline(pd.Timestamp(c), color=MUTED, ls="--", lw=1)
    ax[0].text(pd.Timestamp(c), ax[0].get_ylim()[1] * 0.92, f" {lab}", color=MUTED, fontsize=8)
ax[0].set(title="Reviews per quarter", xlabel="", ylabel="reviews")
df.rating.value_counts().sort_index().plot.bar(ax=ax[1], color=BLUE, width=0.6)
ax[1].set(title="Star ratings", xlabel="stars", ylabel="reviews"); ax[1].tick_params(axis="x", rotation=0)
upc = df.user_id.value_counts()
ax[2].hist(upc.clip(upper=15), bins=np.arange(1, 17) - 0.5, color=BLUE, rwidth=0.8)
ax[2].set_yscale("log"); ax[2].set(title="Reviews per user (capped at 15)", xlabel="reviews", ylabel="users (log)")
viz.save("01_data"); plt.show()
print(f"users with one review: {(upc == 1).mean():.1%}")
""")

# ------------------------------------------------------------------ 2 aspects
md("""
## 2. Aspect extraction
Each review is split into sentences, and each sentence into clauses at contrast words (*but, however,
although, except, ...*). Every clause is then matched against a 12-category aspect lexicon. The
clause split lets "smells lovely but the pump broke" yield a **positive** *scent* mention and a
**negative** *packaging* mention, instead of one mixed sentence.
""")
code("""
example = "Smells lovely and the texture is light, but the pump broke after a week. Worth the price though."
print(aspects.split_clauses(example))
display(aspects.extract_mentions(pd.DataFrame({"review_id": [0], "text": [example]}))[["aspect", "term", "clause"]])
""")
code("""
cache = "data/mentions.parquet"
if os.path.exists(cache):
    mentions = pd.read_parquet(cache)
else:
    mentions = sentiment.score_vader(aspects.extract_mentions(df))   # ~4 min on CPU
    mentions.to_parquet(cache)
print(f"{len(mentions):,} aspect mentions | {mentions.review_id.nunique() / len(df):.1%} of reviews "
      f"mention at least one aspect")
ax = mentions.aspect.value_counts().sort_values().plot.barh(color=BLUE, figsize=(6, 3.8))
ax.set(title="Aspect mentions", xlabel="mentions", ylabel="")
viz.save("02_aspects"); plt.show()
""")
md("Lexicon coverage check: frequent noun-phrase heads (found with spaCy) that the lexicon does not yet cover.")
code("""
try:
    import spacy
    try:
        spacy.load("en_core_web_sm")
    except OSError:
        !python -m spacy download -q en_core_web_sm
    display(aspects.mine_candidate_terms(df.text.sample(3000, random_state=SEED), top_n=30).T)
except Exception as e:
    print("spaCy step skipped:", e)
""")

# ------------------------------------------------------------------ 3 sentiment
md("""
## 3. Aspect-level sentiment
Each mention is scored with VADER (Hutto & Gilbert, 2014) on its own clause, giving a compound score
in [-1, 1].

**Validity check.** Mean aspect sentiment should rise with the star rating.
""")
code("""
r = df.set_index("review_id").rating
per_review = mentions.groupby("review_id").score.mean().to_frame().join(r)
check = per_review.groupby("rating").score.agg(["mean", "std", "count"])
rho = stats.spearmanr(per_review.score, per_review.rating).statistic
display(check.round(3)); print(f"Spearman(aspect sentiment, stars) = {rho:.3f}")

fig, ax = plt.subplots(1, 2, figsize=(12, 3.6))
ax[0].errorbar(check.index, check["mean"], yerr=1.96 * check["std"] / np.sqrt(check["count"]),
               color=BLUE, marker="o", lw=2, capsize=3)
ax[0].set(title="Mean aspect sentiment by star rating", xlabel="stars", ylabel="mean VADER score")
pol = mentions.groupby("aspect").polarity.value_counts(normalize=True).unstack()[[-1, 0, 1]]
pol.columns = ["negative", "neutral", "positive"]
pol.sort_values("negative").plot.barh(stacked=True, ax=ax[1], color=[ORANGE, "#d9d8d2", BLUE], width=0.75)
ax[1].set(title="Polarity share per aspect", xlabel="share of mentions", ylabel="")
ax[1].legend(loc="lower right", fontsize=8, ncol=3)
viz.save("03_sentiment"); plt.show()
""")
md("""
### 3b. Transformer check (runs automatically on a GPU)
A DeBERTa-v3 model fine-tuned for aspect-based sentiment (Yang et al., PyABSA) reads the clause
*together with* the aspect term. On a sample of mentions we measure how often it agrees with VADER.
""")
code("""
import torch
RUN_TRANSFORMER = torch.cuda.is_available()
if RUN_TRANSFORMER:
    sample = mentions.sample(2000, random_state=SEED)
    tf = sentiment.score_transformer(sample, batch_size=32)
    from sklearn.metrics import cohen_kappa_score
    agree = (tf.polarity.values == sample.polarity.values).mean()
    kappa = cohen_kappa_score(tf.polarity.values, sample.polarity.values)
    print(f"VADER vs DeBERTa-ABSA polarity agreement: {agree:.1%} (Cohen's kappa {kappa:.2f})")
    display(pd.crosstab(pd.Series(sample.polarity.values, name="VADER"),
                        pd.Series(tf.polarity.values, name="DeBERTa"), normalize="index").round(2))
    s_rating = sample.review_id.map(r)
    print("Spearman with stars  VADER: %.3f  DeBERTa: %.3f" % (
        stats.spearmanr(sample.score, s_rating).statistic, stats.spearmanr(tf.score, s_rating).statistic))
else:
    print("No GPU: transformer check skipped (Runtime > Change runtime type > T4 GPU to enable).")
""")

code("""
# Free memory before the graph stages: the clause and review text are no longer needed.
import gc
mentions = mentions[["review_id", "aspect", "score", "polarity"]].copy()
df = df.drop(columns=["text"])
hist, target = data.time_split(df, CUTOFF)
for name in ["tf", "sample", "per_review"]:
    globals().pop(name, None)
gc.collect()
print(f"mentions: {mentions.memory_usage(deep=True).sum() / 1e6:.0f} MB | reviews: {df.memory_usage(deep=True).sum() / 1e6:.0f} MB")
""")

# ------------------------------------------------------------------ 4 graph
md("""
## 4. Weighted user–aspect graph and communities (history only)

**Taste profiles.** Raw sentiment mostly reflects *product quality*: everyone praises a good serum.
To group people by *taste*, we first remove the product effect. A user's residual on aspect *a* is
their sentiment minus (global mean on *a* + the product's shrunk offset on *a*). One extra
"overall" dimension holds the star-rating residual.

**Graph construction.**
* **Bipartite user–aspect graph.** Edge weight = number of mentions; edge attribute = mean residual.
* **User–user projection.** Edge weight is
  `w(u,v) = Σ_a min(p_ua, p_va) · (1 − |s_ua − s_va| / 2)`, i.e. shared interest × opinion agreement.
  Each user keeps only its k = 15 strongest neighbours.
* **Communities.** Louvain (Blondel et al., 2008) runs on the *core* users, those with at least 2
  reviews in history.
* **Inductive assignment.** Every other user, including one-review users, joins the community with
  the largest total similarity among their 15 nearest core users.
""")
code("""
mh = mentions[mentions.review_id.isin(hist.review_id)]
mt = mentions[mentions.review_id.isin(target.review_id)]
h_res, mh_res, _ = F.residualize(hist, mh)
profiles = F.taste_mentions(h_res, mh_res)
part, info = graph.build_communities(hist, profiles, target.user_id.unique(), k=K, seed=SEED)
G, core = info["graph"], info["core_part"]
B = graph.bipartite_graph(info["users"], info["aspects"], info["N"], info["S"])
rand_core = graph.random_partition(core, seed=SEED)
sizes = pd.Series(core).value_counts()
print(f"bipartite graph: {B.number_of_nodes():,} nodes, {B.number_of_edges():,} weighted user-aspect edges")
print(f"user graph: {G.number_of_nodes():,} core users, {G.number_of_edges():,} edges, mean degree "
      f"{2 * G.number_of_edges() / G.number_of_nodes():.1f}")
print(f"Louvain: {len(sizes)} communities | modularity Q = {graph.modularity(G, core):.3f} "
      f"(size-matched random partition: {graph.modularity(G, rand_core):.3f})")
print(f"test users with a community: {target.user_id.isin(part.keys()).mean():.1%}  "
      f"({len(part) - len(core):,} assigned inductively)")
print(f"[{time.time() - T0:.0f}s]")
""")
code("""
from sklearn.metrics import adjusted_rand_score
sweep = []
for res in [0.5, 1.0, 1.5, 2.0, 3.0]:
    p = graph.louvain(G, resolution=res, seed=SEED)
    sweep.append({"resolution": res, "communities": len(set(p.values())), "modularity": graph.modularity(G, p)})
sweep = pd.DataFrame(sweep)
order = list(core)
runs = [graph.louvain(G, seed=s) for s in range(5)]
aris = [adjusted_rand_score([runs[i][u] for u in order], [runs[j][u] for u in order])
        for i in range(5) for j in range(i + 1, 5)]
print(f"seed stability: mean ARI between 5 Louvain runs = {np.mean(aris):.3f} (min {np.min(aris):.3f})")

fig, ax = plt.subplots(1, 2, figsize=(12, 3.4))
ax[0].bar(range(len(sizes)), sizes.values, color=BLUE, width=0.8)
ax[0].set(title="Community sizes (core users)", xlabel="community (by size)", ylabel="users")
ax[1].plot(sweep.resolution, sweep.modularity, marker="o", color=BLUE, lw=2)
for _, row in sweep.iterrows():
    ax[1].annotate(f"{int(row.communities)}", (row.resolution, row.modularity), textcoords="offset points",
                   xytext=(0, 7), ha="center", fontsize=8, color=MUTED)
ax[1].set(title="Resolution sweep (labels = number of communities)", xlabel="Louvain resolution",
          ylabel="modularity Q")
viz.save("04_communities"); plt.show()
""")
code("""
# What each community is about: mean taste residual per aspect (rows = 12 largest communities)
users = info["users"]; Pm = pd.DataFrame(info["P"], index=users, columns=info["aspects"])
Sm = pd.DataFrame(info["S"], index=users, columns=info["aspects"])
top = sizes.index[:12]
lab = pd.Series(core)
prof_s = pd.DataFrame({c: (Sm.loc[lab[lab == c].index] * Pm.loc[lab[lab == c].index]).sum()
                         / Pm.loc[lab[lab == c].index].sum() for c in top}).T
prof_i = pd.DataFrame({c: Pm.loc[lab[lab == c].index].mean() for c in top}).T
fig, ax = plt.subplots(1, 2, figsize=(14, 4.4))
lim = np.nanmax(np.abs(prof_s.values))
im = ax[0].imshow(prof_s.values, cmap=viz.DIVERGING, vmin=-lim, vmax=lim, aspect="auto")
ax[0].set_title("Taste: mean residual opinion"); plt.colorbar(im, ax=ax[0], fraction=0.04)
im = ax[1].imshow(prof_i.values, cmap="Blues", aspect="auto")
ax[1].set_title("Interest: share of mentions"); plt.colorbar(im, ax=ax[1], fraction=0.04)
for a in ax:
    a.set_xticks(range(prof_s.shape[1])); a.set_xticklabels(prof_s.columns, rotation=45, ha="right")
    a.set_yticks(range(len(top))); a.set_yticklabels([f"C{c} ({sizes[c]})" for c in top]); a.grid(False)
viz.save("04_community_profiles"); plt.show()
""")
code("""
# Community-level graph: node = community (size = users), edge = total similarity between communities
big = sizes[sizes >= 50].index
CG = nx.Graph()
for c in big: CG.add_node(c, size=sizes[c])
for u, v, w in G.edges(data="weight"):
    a, b = core[u], core[v]
    if a != b and a in big and b in big:
        CG.add_edge(a, b, weight=CG[a][b]["weight"] + w if CG.has_edge(a, b) else w)
pos = nx.spring_layout(CG, weight="weight", seed=SEED)
plt.figure(figsize=(7.5, 6.2))
ew = np.array([d["weight"] for *_, d in CG.edges(data=True)])
nx.draw_networkx_edges(CG, pos, width=0.3 + 5 * ew / ew.max(), edge_color="#c9c8c2")
nx.draw_networkx_nodes(CG, pos, node_size=[CG.nodes[c]["size"] / 2 for c in CG], node_color=BLUE)
nx.draw_networkx_labels(CG, pos, labels={c: f"C{c}" for c in CG}, font_color="white", font_size=7)
plt.title("Taste communities (node area = users, edge width = cross-community similarity)")
plt.axis("off"); viz.save("04_community_graph"); plt.show()
""")

# ------------------------------------------------------------------ 5 community vs overall
md("""
## 5. Community-specific vs overall sentiment: in-sample and cross-fitted

For every large community and every aspect, we compare the community's share of positive mentions
against everyone else's, using a two-proportion z-test with Benjamini–Hochberg correction.

**(a) In-sample.** Communities are formed and then tested on the *same* reviews. This is how such
analyses are usually reported.

**(b) Cross-fitted.** Each user's history is split at random into fold A and fold B. Communities are
formed on A and tested on B, so a difference only counts if it shows up in reviews that played no part
in forming the community.

Both versions are repeated with a size-matched random partition. If the real communities do no better
than random ones in (b), the in-sample "differences" were produced by forming the groups and testing
them on the same data.
""")
code("""
def shift_table(m_eval, reviews_eval, partition, n_top=10):
    tagged = analysis.tag_mentions(m_eval, reviews_eval, partition)
    top_c = pd.Series(partition).value_counts().index[:n_top]
    return analysis.aspect_sentiment_shift(tagged, top_c, min_mentions=30), tagged

# (a) in-sample: communities from all of history, tested on all of history
shift_in, tagged_in = shift_table(mh, hist, core)
shift_in_rand, _ = shift_table(mh, hist, graph.random_partition(core, seed=SEED))

# (b) cross-fitted: communities from fold A, tested on fold B
multi = hist[hist.user_id.map(hist.user_id.value_counts()) >= 2]
fold = analysis.split_folds(multi, seed=SEED)
hA, hB = multi[fold == "A"], multi[fold == "B"]
mA, mB = mh[mh.review_id.isin(hA.review_id)], mh[mh.review_id.isin(hB.review_id)]
hA_res, mA_res, _ = F.residualize(hA, mA)
part_A, _ = graph.build_communities(hA, F.taste_mentions(hA_res, mA_res), [], k=K, min_core_reviews=1, seed=SEED)
shift_cf, _ = shift_table(mB, hB, part_A)
shift_cf_rand, _ = shift_table(mB, hB, graph.random_partition(part_A, seed=SEED))

cal = pd.DataFrame({"in-sample, real communities": analysis.pvalue_calibration(shift_in),
                    "in-sample, random partition": analysis.pvalue_calibration(shift_in_rand),
                    "cross-fitted, real communities": analysis.pvalue_calibration(shift_cf),
                    "cross-fitted, random partition": analysis.pvalue_calibration(shift_cf_rand)}).T
display(cal.round(3))
del multi, hA, hB, mA, mB, hA_res, mA_res; gc.collect()
print(f"[{time.time() - T0:.0f}s]")
""")
code("""
fig, ax = plt.subplots(1, 3, figsize=(15, 3.8), gridspec_kw={"width_ratios": [1, 1, 1.5]})
bins = np.linspace(0, 1, 21)
for a, (real, rnd, title) in zip(ax[:2], [(shift_in, shift_in_rand, "(a) In-sample"),
                                          (shift_cf, shift_cf_rand, "(b) Cross-fitted")]):
    a.hist(rnd.p, bins=bins, color=GREY, alpha=0.6, label="random partition", density=True)
    a.hist(real.p, bins=bins, histtype="step", lw=2.2, color=BLUE, label="real communities", density=True)
    a.axhline(1, color=MUTED, ls=":", lw=1)
    a.set(title=f"{title}: p-value distribution", xlabel="p-value", ylabel="density"); a.legend(fontsize=8)
heat = analysis.heatmap_table(tagged_in, pd.Series(core).value_counts().index[:10])
lim = np.nanmax(np.abs(heat.values))
im = ax[2].imshow(heat.values, cmap=viz.DIVERGING, vmin=-lim, vmax=lim, aspect="auto")
ax[2].set_xticks(range(heat.shape[1])); ax[2].set_xticklabels(heat.columns, rotation=45, ha="right")
ax[2].set_yticks(range(len(heat))); ax[2].set_yticklabels([f"C{c}" for c in heat.index]); ax[2].grid(False)
ax[2].set_title("In-sample: community minus overall sentiment"); plt.colorbar(im, ax=ax[2], fraction=0.04)
viz.save("05_community_vs_overall"); plt.show()
""")

# ------------------------------------------------------------------ 6 forecasting
md("""
## 6. Does community information improve sentiment prediction?

**Task (forecasting).** For a user who is about to review product *p*, predict **before the review is
written**: (i) the star rating and (ii) the sentiment they will express on each aspect they mention.
This isolates what the *community* knows, because the text of the new review is never seen.

**Proposed model: CAP (Community-Anchored Partial pooling).** A hierarchical baseline predictor:

`ŝ(u,p,a) = μ_a + b_pa + b_ca + b_cpa + b_ua`

* `μ_a` is the global mean sentiment on aspect *a*.
* `b_pa` is product *p*'s shrunk offset on aspect *a*.
* `b_ca` is the community's offset on aspect *a*. It comes only from *other* members' residuals.
* `b_cpa` is the community's opinion of *this* product.
* `b_ua` is the user's own residual *after* the community offset. A user with little history therefore
  falls back to the community, and one with long history overrides it.

**Cross-fitting (the fix for section 5's circularity).** The history is split per user into fold A
and fold B. Communities are formed on A, and their offsets (`b_ca`, `b_cpa`) are estimated only on B.
We also report a **naive** CAP whose communities are formed and scored on the same reviews.

Every level is shrunk as `sum / (n + λ)`. The λ values are tuned on the 2020 validation period.

**Baselines.** Product-only; +user (personalisation, the strongest standard baseline); +community.

**Learned variant.** Gradient boosting on the same statistics, trained on the validation period.
""")
code("""
def forecasting_stage(cutoff, end=None, cross_fit=True):
    \"\"\"History before `cutoff`, targets after it. With cross_fit, communities are formed on fold A of
    the history and their offsets estimated on fold B, so no review both forms and scores a community.\"\"\"
    h, tg = data.time_split(df, cutoff)
    if end: tg = tg[tg.date < end]
    m_h, m_t = mentions[mentions.review_id.isin(h.review_id)], mentions[mentions.review_id.isin(tg.review_id)]
    rows = None
    if cross_fit:
        fold = analysis.split_folds(h, seed=SEED)
        h_form, rows = h[fold == "A"], set(h.review_id[fold == "B"])
    else:
        h_form = h
    m_form = m_h[m_h.review_id.isin(h_form.review_id)]
    r_h, r_m, _ = F.residualize(h_form, m_form)
    p, _ = graph.build_communities(h_form, F.taste_mentions(r_h, r_m), tg.user_id.unique(), k=K, seed=SEED)
    return F.compute_stats(h, m_h, tg, m_t, p, community_rows=rows), (h, m_h, tg, m_t, p)

# validation (2020) -> tune shrinkage; test (2021+) -> evaluate. Both cross-fitted (proposed) and in-sample (naive).
(Tv, MTv), val_parts = forecasting_stage(CUTOFF_VAL, CUTOFF, cross_fit=True)
(T, MT), _ = forecasting_stage(CUTOFF, cross_fit=True)
(Tv_n, MTv_n), _ = forecasting_stage(CUTOFF_VAL, CUTOFF, cross_fit=False)
(T_n, MT_n), _ = forecasting_stage(CUTOFF, cross_fit=False)
lam_a, grid_a = F.tune_lambdas(Tv, MTv, target="aspect")
lam_r, grid_r = F.tune_lambdas(Tv, MTv, target="rating")
lam_an, _ = F.tune_lambdas(Tv_n, MTv_n, target="aspect")
lam_rn, _ = F.tune_lambdas(Tv_n, MTv_n, target="rating")
print("tuned shrinkage (cross-fitted)  aspect:", lam_a, " rating:", lam_r)

pa, pr = F.predict(T, MT, lam_a), F.predict(T, MT, lam_r)
pan, prn = F.predict(T_n, MT_n, lam_an), F.predict(T_n, MT_n, lam_rn)
preds = {"rating": dict(pr["rating"]), "aspect": dict(pa["aspect"])}
preds["rating"]["CAP naive (in-sample)"] = prn["rating"]["CAP (full)"]
preds["aspect"]["CAP naive (in-sample)"] = pan["aspect"]["CAP (full)"]
results = F.evaluate(T, MT, preds)
print(f"test: {len(T):,} reviews, {len(MT):,} aspect mentions   [{time.time() - T0:.0f}s]")
display(results[results.users == "all"].drop(columns="users").round(4))
""")
code("""
fig, ax = plt.subplots(1, 2, figsize=(13, 3.8))
models = ["product", "+user", "+community", "CAP (full)"]
colors = [GREY, BLUE, ORANGE, AQUA]
for a, level in zip(ax, ["rating", "aspect"]):
    sub = results[(results.level == level) & (results.users != "all")]
    groups = sub.users.unique(); w = 0.2
    for i, (mname, col) in enumerate(zip(models, colors)):
        vals = sub[sub.model == mname].set_index("users").loc[groups, "auc"]
        a.bar(np.arange(len(groups)) + (i - 1.5) * w, vals, w * 0.92, color=col, label=mname)
    a.axhline(0.5, color=MUTED, lw=1, ls=":")
    a.set_xticks(range(len(groups))); a.set_xticklabels([f"{g}\\n(n={sub[sub.users == g].n.iloc[0]:,})" for g in groups])
    lo = sub.auc.min() - 0.03
    a.set_ylim(max(0.45, lo), sub.auc.max() + 0.02)
    a.set(title=f"Forecasting {'star rating' if level == 'rating' else 'aspect sentiment'}: AUC by user history",
          ylabel="AUC (positive vs negative)")
ax[0].legend(fontsize=8, ncol=2)
viz.save("06_forecasting_strata"); plt.show()
""")
md("Paired bootstrap over **users** (500 resamples): the gain from adding community information on top of the user's own history, and on top of the product baseline.")
code("""
rows = []
for level, frame, y in [("rating", T, "rating"), ("aspect", MT, "score")]:
    b = F.history_bucket(frame.u_n.to_numpy())
    for grp in ["all"] + list(b.cat.categories):
        mask = np.ones(len(frame), bool) if grp == "all" else (b == grp).to_numpy()
        fr = frame[mask]
        for a_, b_ in [("CAP (full)", "+user"), ("+community", "product"), ("CAP (full)", "CAP naive (in-sample)")]:
            ci = F.cluster_bootstrap(fr, fr[y], preds[level][a_][mask], preds[level][b_][mask],
                                     metric="auc", level=level, n_boot=500, seed=SEED)
            rows.append({"level": level, "users": grp, "comparison": f"{a_} vs {b_}",
                         "auc_gain": ci[1], "ci_low": ci[0], "ci_high": ci[2]})
boot = pd.DataFrame(rows)
display(boot.round(4))

fig, ax = plt.subplots(1, 3, figsize=(17, 3.6), sharey=True)
for a, comp in zip(ax, ["+community vs product", "CAP (full) vs +user", "CAP (full) vs CAP naive (in-sample)"]):
    sub = boot[boot.comparison == comp].reset_index(drop=True)
    ylab = [f"{r.level} · {r.users}" for r in sub.itertuples()]
    a.errorbar(sub.auc_gain, range(len(sub)), xerr=[sub.auc_gain - sub.ci_low, sub.ci_high - sub.auc_gain],
               fmt="o", color={"+community vs product": BLUE, "CAP (full) vs +user": AQUA}.get(comp, ORANGE),
               capsize=3, lw=1.6)
    a.axvline(0, color=MUTED, lw=1)
    a.set_yticks(range(len(sub))); a.set_yticklabels(ylab); a.invert_yaxis()
    a.set(title=f"AUC gain: {comp} (95% CI)", xlabel="AUC difference")
    a.xaxis.set_major_locator(plt.MaxNLocator(5))
viz.save("06_bootstrap"); plt.show()
""")
code("""
# learned variant: gradient boosting trained on the 2020 validation period
gbm_rows, gbm_models = [], {}
for level, tr, te, y in [("rating", Tv, T, "rating"), ("aspect", MTv, MT, "score")]:
    b = F.history_bucket(te.u_n.to_numpy())
    for name, which in F.GBM_SETS.items():
        p, model = F.fit_gbm(tr, te, level, which, seed=SEED)
        gbm_models[(level, name)] = model
        row = {"level": level, "model": name, "rmse": F.rmse(te[y], p), "auc": F.auc_pos_neg(te[y], p, level)}
        for c in b.cat.categories:
            mk = (b == c).to_numpy(); row[f"auc {c}"] = F.auc_pos_neg(te[y][mk], p[mk], level)
        gbm_rows.append(row)
gbm = pd.DataFrame(gbm_rows)
display(gbm.round(4))
""")
code("""
from sklearn.inspection import permutation_importance
model = gbm_models[("rating", "GBM +user+community")]
X = F.gbm_features(T, "rating", "user community")
sub = np.random.default_rng(SEED).choice(len(X), min(6000, len(X)), replace=False)
imp = permutation_importance(model, X.iloc[sub], T.rating.iloc[sub], n_repeats=5, random_state=SEED,
                             scoring="neg_root_mean_squared_error")
imp = pd.Series(imp.importances_mean, index=X.columns).sort_values()
is_c = [c.startswith(("c_", "cp_")) for c in imp.index]
plt.figure(figsize=(6.5, 3.6))
plt.barh(imp.index, imp.values, color=[ORANGE if c else BLUE for c in is_c])
plt.title("Permutation importance, GBM +user+community (rating)\\norange = community features")
plt.xlabel("increase in RMSE when shuffled"); viz.save("06_gbm_importance"); plt.show()
""")

# ------------------------------------------------------------------ 7 ablation
md("""
## 7. Ablation: does the way communities are built matter?

We compare five partitions of the same users, scored with the same CAP model. All five use
in-sample offsets, so they are compared under identical conditions:

| Partition | How it is built |
|---|---|
| **taste** (proposed) | Residual profiles, interest × agreement |
| **raw** | Raw sentiment, interest × agreement |
| **interest-only** | Interest overlap only |
| **k-means** | The same taste profiles clustered with k-means, with as many clusters as Louvain found |
| **random** | Size-matched random partition |
""")
code("""
abl_parts = {"taste (proposed)": part}
p_raw, _ = graph.build_communities(hist, mh, target.user_id.unique(), k=K, seed=SEED)
p_int, _ = graph.build_communities(hist, profiles, target.user_id.unique(), k=K, agreement=False, seed=SEED)
abl_parts["raw sentiment"] = p_raw
abl_parts["interest-only"] = p_int
abl_parts["k-means"] = graph.kmeans_partition(info["users"], info["P"], info["S"], len(sizes), seed=SEED)
abl_parts["random"] = graph.random_partition(part, seed=SEED)
abl = []
for name, pk in abl_parts.items():
    a_T, a_MT = F.compute_stats(hist, mh, target, mt, pk)
    q_r, q_a = F.predict(a_T, a_MT, lam_r), F.predict(a_T, a_MT, lam_a)
    abl.append({"partition": name, "communities": len(set(pk.values())),
                "rating AUC +community": F.auc_pos_neg(a_T.rating, q_r["rating"]["+community"], "rating"),
                "rating AUC CAP": F.auc_pos_neg(a_T.rating, q_r["rating"]["CAP (full)"], "rating"),
                "aspect AUC +community": F.auc_pos_neg(a_MT.score, q_a["aspect"]["+community"], "aspect"),
                "aspect AUC CAP": F.auc_pos_neg(a_MT.score, q_a["aspect"]["CAP (full)"], "aspect")})
abl = pd.DataFrame(abl)
display(abl.round(4))
user_only = {"rating": F.auc_pos_neg(T.rating, pr["rating"]["+user"], "rating"),
             "aspect": F.auc_pos_neg(MT.score, pa["aspect"]["+user"], "aspect")}

fig, ax = plt.subplots(1, 2, figsize=(13, 3.4), sharey=True)
for a, level in zip(ax, ["rating", "aspect"]):
    y = np.arange(len(abl))
    a.scatter(abl[f"{level} AUC +community"], y, color=ORANGE, s=60, label="+community", zorder=3)
    a.scatter(abl[f"{level} AUC CAP"], y, color=AQUA, s=60, label="CAP (community+user)", zorder=3)
    a.axvline(user_only[level], color=BLUE, lw=1.5, ls="--", label="+user (no community)")
    a.set_yticks(y); a.set_yticklabels(abl.partition); a.invert_yaxis()
    a.set(title=f"{level.capitalize()} forecasting by partition", xlabel="AUC")
ax[0].legend(fontsize=8, loc="upper left")
viz.save("07_ablation"); plt.show()
print(f"[{time.time() - T0:.0f}s]")
""")

# ------------------------------------------------------------------ 8 attention
md("""
## 8. Secondary task: forecasting *which* aspects a user will talk about

Communities may capture *what people care about* rather than *how they feel*. Here we predict the set
of aspects mentioned in the next review, a 12-way multi-label task. The pooling hierarchy is the same
as before: product → community → user.
""")
code("""
s_v = A.attention_stats(*val_parts)
best = min(((lc, lu) for lc in [1, 3, 10, 30, 100] for lu in [1, 3, 10, 30, 100]),
           key=lambda x: A.attention_scores(s_v["Y"], A.attention_predict(s_v, *x)["CAP (community+user)"])[1])
s_t = A.attention_stats(hist, mh, target, mt, part)
att_pred = A.attention_predict(s_t, *best)
bk = F.history_bucket(s_t["u_n"])
att = []
for name, p in att_pred.items():
    auc, ll = A.attention_scores(s_t["Y"], p)
    row = {"model": name, "mean AUC": auc, "log-loss": ll}
    for c in bk.cat.categories:
        row[f"AUC {c}"] = A.attention_scores(s_t["Y"], p, (bk == c).to_numpy())[0]
    att.append(row)
att = pd.DataFrame(att); print("tuned (lam_c, lam_u):", best); display(att.round(4))
ax = att.set_index("model")[[c for c in att.columns if c.startswith("AUC ")]].T.plot.bar(
    figsize=(8, 3.4), color=[GREY, BLUE, ORANGE, AQUA], width=0.8)
ax.set(title="Aspect-attention forecasting: mean per-aspect AUC by user history", ylabel="AUC",
       ylim=(0.5, att.filter(like="AUC").values.max() + 0.02)); ax.tick_params(axis="x", rotation=0)
ax.legend(fontsize=8, ncol=4); viz.save("08_attention"); plt.show()
""")

# ------------------------------------------------------------------ 9 summary
md("## 9. Summary of findings and saved outputs")
code("""
allr = results[results.users == "all"].set_index(["level", "model"]).auc
summary = pd.DataFrame([
    ("Communities (core users)", f"{len(sizes)}, modularity {graph.modularity(G, core):.3f} vs random {graph.modularity(G, rand_core):.3f}"),
    ("Louvain seed stability (ARI)", f"{np.mean(aris):.3f}"),
    ("In-sample significant community-aspect shifts (BH)", f"{cal.loc['in-sample, real communities', 'bh_sig']:.1%} vs random {cal.loc['in-sample, random partition', 'bh_sig']:.1%}"),
    ("Cross-fitted significant shifts (BH)", f"{cal.loc['cross-fitted, real communities', 'bh_sig']:.1%} vs random {cal.loc['cross-fitted, random partition', 'bh_sig']:.1%}"),
    ("Rating AUC: product / +community / +user / CAP", " / ".join(f"{allr['rating', m]:.3f}" for m in ["product", "+community", "+user", "CAP (full)"])),
    ("Aspect AUC: product / +community / +user / CAP", " / ".join(f"{allr['aspect', m]:.3f}" for m in ["product", "+community", "+user", "CAP (full)"])),
    ("GBM rating AUC: +user / +user+community", f"{gbm.set_index(['level','model']).auc['rating','GBM +user']:.3f} / {gbm.set_index(['level','model']).auc['rating','GBM +user+community']:.3f}"),
    ("Attention AUC: +user / CAP", f"{att.set_index('model')['mean AUC']['+user']:.3f} / {att.set_index('model')['mean AUC']['CAP (community+user)']:.3f}"),
], columns=["finding", "value"])
display(summary)
os.makedirs("results", exist_ok=True)
for name, frame in {"summary": summary, "forecasting": results, "bootstrap": boot, "gbm": gbm,
                    "ablation": abl, "attention": att, "pvalue_calibration": cal.reset_index(),
                    "shift_in_sample": shift_in, "shift_cross_fitted": shift_cf, "resolution_sweep": sweep}.items():
    frame.to_csv(f"results/{name}.csv", index=False)
print(sorted(os.listdir("results/figures")))
print(f"total runtime {time.time() - T0:.0f}s")
if IN_COLAB:
    !zip -qr results.zip results
    from google.colab import files; files.download("results.zip")
""")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3"},
                  "colab": {"provenance": []}, "accelerator": "GPU",
                  "language_info": {"name": "python"}}
nbf.write(nb, "notebooks/community_aware_absa.ipynb")
print("wrote notebooks/community_aware_absa.ipynb")
