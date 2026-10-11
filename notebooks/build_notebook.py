"""Generates community_aware_absa.ipynb from the cells below (keeps the notebook diff-able)."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md("""
# Community-Aware Aspect-Based Sentiment Analysis

**Team:** Munal Singh (24MID0095), Devadharsh S (24MID0210)

Pipeline: Amazon reviews -> aspect extraction -> aspect-level sentiment ->
weighted user-aspect graph -> user-user similarity graph -> Louvain communities ->
community vs overall sentiment -> does community information improve sentiment prediction?

Runs on a free Colab CPU runtime in about 5 minutes. A GPU runtime is only needed for
the optional transformer ABSA comparison (section 4b).
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
else:
    if os.path.basename(os.getcwd()) == "notebooks":
        os.chdir("..")
sys.path.insert(0, "src")
os.makedirs("results/figures", exist_ok=True)
print("working dir:", os.getcwd())
""")
code("""
import numpy as np, pandas as pd, matplotlib.pyplot as plt, networkx as nx
from cabsa import data, aspects, sentiment, graph, analysis, experiment

CATEGORY = "All_Beauty"   # Amazon Reviews 2023 category
MIN_REVIEWS = 3           # users need >= 3 reviews to have a profile + a test review
K_NEIGHBOURS = 15         # sparsification of the user-user graph
RESOLUTION = 1.0          # Louvain resolution
SEED = 42
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False})
def savefig(name):
    plt.tight_layout(); plt.savefig(f"results/figures/{name}.png", dpi=200, bbox_inches="tight")
""")

md("""
## 1. Data
Amazon Reviews 2023 (McAuley Lab, Hou et al., 2024), category *All_Beauty*. We keep users with at
least 3 reviews, then hold out each user's **most recent** review as the test review. Everything in
sections 3-6 is built from training reviews only.
""")
code("""
path = data.download(CATEGORY, "data")
df = data.temporal_split(data.load_reviews(path, min_reviews=MIN_REVIEWS))
print(f"{len(df):,} reviews | {df.user_id.nunique():,} users | {df.parent_asin.nunique():,} products")
print(df.split.value_counts().to_dict())
df.head(3)[["user_id", "rating", "text", "split"]]
""")
code("""
fig, ax = plt.subplots(1, 2, figsize=(10, 3.2))
df.rating.value_counts().sort_index().plot.bar(ax=ax[0], color="0.35")
ax[0].set(title="Star rating distribution", xlabel="stars", ylabel="reviews")
df.user_id.value_counts().clip(upper=20).value_counts().sort_index().plot.bar(ax=ax[1], color="0.35")
ax[1].set(title="Reviews per user (capped at 20)", xlabel="reviews", ylabel="users")
savefig("01_data"); plt.show()
""")

md("""
## 2. Aspect extraction
Reviews are split into sentences, then into clauses at contrast words (*but, however, although...*),
and each clause is matched against a 12-category aspect lexicon. Clause splitting keeps
"smells lovely but the pump broke" as two opposite opinions instead of one mixed sentence.
""")
code("""
mentions = aspects.extract_mentions(df)
cov = mentions.review_id.nunique() / len(df)
print(f"{len(mentions):,} aspect mentions | {cov:.1%} of reviews mention at least one aspect")
mentions.aspect.value_counts().plot.barh(color="0.35", figsize=(6, 3.5), title="Aspect mentions")
plt.gca().invert_yaxis(); savefig("02_aspects"); plt.show()
mentions.sample(8, random_state=SEED)[["aspect", "term", "clause"]]
""")
md("Lexicon expansion: frequent noun-phrase heads (spaCy) that the lexicon does not yet cover.")
code("""
try:
    import spacy
    try:
        spacy.load("en_core_web_sm")
    except OSError:
        !python -m spacy download -q en_core_web_sm
    cand = aspects.mine_candidate_terms(df.text.sample(3000, random_state=SEED), top_n=40)
    display(cand.T)
except Exception as e:
    print("spaCy step skipped:", e)
""")

md("""
## 3. Aspect-level sentiment
Each mention is scored with VADER on its own clause (compound score in [-1, 1]).
Check: the mean aspect sentiment of a review should rise with its star rating.
""")
code("""
mentions = sentiment.score_vader(mentions)
check = sentiment.review_level_check(mentions, df)
display(check.round(3))
r = df.set_index("review_id").rating
rho = mentions.groupby("review_id").score.mean().to_frame().join(r).corr(method="spearman").iloc[0, 1]
print(f"Spearman(aspect sentiment, stars) = {rho:.3f}")
check["mean"].plot(marker="o", color="0.2", yerr=check["std"] / np.sqrt(check["count"]), capsize=3,
                   title="Mean aspect sentiment by star rating", figsize=(5, 3))
savefig("03_sentiment_check"); plt.show()
""")
code("""
pol = mentions.groupby("aspect").polarity.value_counts(normalize=True).unstack()[[-1, 0, 1]]
pol.columns = ["negative", "neutral", "positive"]
pol.sort_values("negative").plot.barh(stacked=True, color=["0.15", "0.6", "0.85"], figsize=(7, 4),
                                      title="Overall polarity per aspect")
plt.legend(loc="lower right", fontsize=8); savefig("03_polarity"); plt.show()
""")

md("""
### 3b. (Optional, GPU) Transformer ABSA vs VADER
A DeBERTa-v3 model fine-tuned for aspect-based sentiment reads the clause *and* the aspect term.
We score a sample and measure agreement with VADER. Set `RUN_TRANSFORMER = True` on a GPU runtime.
""")
code("""
RUN_TRANSFORMER = False
if RUN_TRANSFORMER:
    sample = mentions.sample(3000, random_state=SEED)
    tf = sentiment.score_transformer(sample)
    agree = (tf.polarity.values == sample.polarity.values).mean()
    print(f"polarity agreement VADER vs DeBERTa-ABSA: {agree:.1%}")
    display(pd.crosstab(sample.polarity.values, tf.polarity.values,
                        rownames=["VADER"], colnames=["DeBERTa"]))
""")

md("""
## 4. Weighted user-aspect graph and user communities
* Bipartite graph: user -- aspect, weight = mention count, attribute = mean opinion.
* Projection: `w(u,v) = sum_a min(p_ua, p_va) * (1 - |s_ua - s_va| / 2)` -- shared interest
  multiplied by opinion agreement; each user keeps the k strongest neighbours.
* Louvain community detection on the projection.
""")
code("""
train_ids = set(df.review_id[df.split == "train"])
m_train = mentions[mentions.review_id.isin(train_ids)]
users, aspect_names, P, S, N = graph.user_aspect_profiles(m_train, df)
B = graph.bipartite_graph(users, aspect_names, N, S)
print(f"bipartite graph: {B.number_of_nodes():,} nodes, {B.number_of_edges():,} weighted user-aspect edges")

nbr_idx, nbr_w = graph.similarity_knn(P, S, k=K_NEIGHBOURS)
G = graph.user_graph(users, nbr_idx, nbr_w)
print(f"user graph: {G.number_of_nodes():,} users, {G.number_of_edges():,} edges, "
      f"density {nx.density(G):.5f}")
part = graph.louvain(G, resolution=RESOLUTION, seed=SEED)
rand = graph.random_partition(part, seed=SEED)
sizes = pd.Series(part).value_counts()
print(f"{len(sizes)} communities | modularity Q = {graph.modularity(G, part):.3f} "
      f"(size-matched random partition: {graph.modularity(G, rand):.3f})")
sizes.head(15).plot.bar(color="0.35", figsize=(7, 3), title="Community sizes (largest 15)")
savefig("04_community_sizes"); plt.show()
""")
md("Robustness: resolution sweep and agreement between Louvain runs with different seeds.")
code("""
from sklearn.metrics import adjusted_rand_score
rows = []
for res in [0.5, 1.0, 1.5, 2.0]:
    p = graph.louvain(G, resolution=res, seed=SEED)
    rows.append({"resolution": res, "communities": len(set(p.values())),
                 "communities_30plus": int((pd.Series(p).value_counts() >= 30).sum()),
                 "modularity": graph.modularity(G, p)})
display(pd.DataFrame(rows).round(3))
order = list(part)
runs = [graph.louvain(G, resolution=RESOLUTION, seed=s) for s in range(5)]
aris = [adjusted_rand_score([runs[i][u] for u in order], [runs[j][u] for u in order])
        for i in range(5) for j in range(i + 1, 5)]
print(f"seed stability: mean ARI between 5 Louvain runs = {np.mean(aris):.3f} (min {np.min(aris):.3f})")
""")
code("""
# community-level view: one node per community (size = members), edge = total cross weight
big = sizes[sizes >= 30].index
CG = nx.Graph()
for c in big: CG.add_node(c, size=sizes[c])
for u, v, w in G.edges(data="weight"):
    a, b = part[u], part[v]
    if a != b and a in big and b in big:
        CG.add_edge(a, b, weight=CG[a][b]["weight"] + w if CG.has_edge(a, b) else w)
pos = nx.spring_layout(CG, weight="weight", seed=SEED)
plt.figure(figsize=(7, 6))
ew = np.array([d["weight"] for *_, d in CG.edges(data=True)])
nx.draw_networkx_edges(CG, pos, width=0.3 + 4 * ew / ew.max(), edge_color="0.7")
nx.draw_networkx_nodes(CG, pos, node_size=[CG.nodes[c]["size"] for c in CG], node_color="0.35")
nx.draw_networkx_labels(CG, pos, font_color="white", font_size=8)
plt.title("Communities (node size = users, edge width = cross-community similarity)")
plt.axis("off"); savefig("04_community_graph"); plt.show()
""")

md("""
## 5. Community-specific sentiment vs overall sentiment
For every large community: what it talks about (Jensen-Shannon divergence of its aspect-interest
distribution from the population; signature aspects by lift) and how its sentiment on each aspect
differs from everyone else (two-proportion z-test, Benjamini-Hochberg corrected).
""")
code("""
mt = analysis.tag_mentions(m_train, df, part)
summary = analysis.community_summary(mt, part, min_users=30)
display(summary.round(3))
top = summary.community.head(10).tolist()
heat = analysis.heatmap_table(mt, top)
plt.figure(figsize=(9, 4.5))
lim = np.nanmax(np.abs(heat.values))
plt.imshow(heat.values, cmap="RdBu", vmin=-lim, vmax=lim, aspect="auto")
plt.xticks(range(heat.shape[1]), heat.columns, rotation=45, ha="right"); plt.yticks(range(len(top)), top)
plt.colorbar(label="community mean - global mean (VADER)"); plt.ylabel("community")
plt.title("Aspect sentiment deviation from overall sentiment"); savefig("05_heatmap"); plt.show()
""")
code("""
shift = analysis.aspect_sentiment_shift(mt, top, min_mentions=30)
print(f"{shift.significant.sum()} of {len(shift)} (community, aspect) cells differ significantly "
      f"from the rest of the population (BH q < 0.05)")
display(shift.sort_values("p").head(15).round(4))
""")

md("""
## 6. Does community information improve sentiment prediction?
Task: predict whether each user's held-out latest review is positive (4-5 stars) or negative
(1-2 stars). Same TF-IDF + logistic regression for every model; only the appended prior features
change. `+random` uses a size-matched random partition as a control. Gains are paired-bootstrap
95% CIs of macro-F1 against text only.
""")
code("""
res, ex = experiment.run(df, mentions, part, rand, n_boot=1000, seed=SEED)
display(res.round(4))
print("Short reviews (<= 12 words):", int(ex["short"].sum()))
print("\\nPriors only (no text at all):")
display(ex["priors_only"].round(4))
""")
code("""
fig, ax = plt.subplots(1, 2, figsize=(10, 3.2))
res.set_index("model")["macro_f1"].plot.barh(ax=ax[0], color="0.35", xlim=(res.macro_f1.min() - 0.01, res.macro_f1.max() + 0.005))
ax[0].set_title("Macro-F1 with text"); ax[0].invert_yaxis()
ex["priors_only"].set_index("model")["auc"].plot.barh(ax=ax[1], color="0.35", xlim=(0.45, 0.75))
ax[1].axvline(0.5, color="k", lw=0.8, ls="--"); ax[1].set_title("AUC from priors alone"); ax[1].invert_yaxis()
savefig("06_prediction"); plt.show()
""")

md("## 7. Save outputs")
code("""
summary.to_csv("results/community_summary.csv", index=False)
shift.to_csv("results/aspect_sentiment_shift.csv", index=False)
res.to_csv("results/prediction_results.csv", index=False)
ex["priors_only"].to_csv("results/priors_only.csv", index=False)
print(sorted(os.listdir("results/figures")))
if IN_COLAB:
    !zip -qr results.zip results
    from google.colab import files; files.download("results.zip")
""")

nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3"},
                  "colab": {"provenance": []}, "language_info": {"name": "python"}}
nbf.write(nb, "notebooks/community_aware_absa.ipynb")
print("wrote notebooks/community_aware_absa.ipynb")
