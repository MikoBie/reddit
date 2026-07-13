# %%
import json
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from sklearn.decomposition import PCA

# %%
## Define globals
ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"
PROC = DATA / "processed"
PNG = ROOT / "png"
COLORS = {"yellow": "#E6B830", "blue": "#A5C9E6", "green": "#73C0C1"}


# %%
## Load data
df = pd.read_csv(PROC / "food_texts.csv", index_col=None)

embeddings_lst = []
for line in open(PROC / "food_texts_embeddings.jsonl", "r"):
    tmp = json.loads(line)
    embeddings_lst.append({"id": tmp["id"], "embeddings": tmp["embeddings"]})

df_embeddings = pd.DataFrame.from_dict(embeddings_lst)
df = pd.merge(df, df_embeddings, on="id")
# %%
## Compute PCA on nromalized embeddings
df_lst = []
for _, tmp in df.groupby("country"):
    vectors = np.array(tmp["embeddings"].tolist())
    matrix = np.vstack(vectors)
    # Calculate Euclidean distance of embeddings
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix /= norms
    ## Compute PCA
    pca = PCA(n_components=2, random_state=8710).fit_transform(matrix)
    pca_df = pd.DataFrame(pca, columns=["Component A", "Component B"])
    tmp = tmp.reset_index(drop=True).join(pca_df)
    df_lst.append(tmp)

df = pd.concat(df_lst)

# %%
## MOTIVATION
df_mot = df.query("mot_refl > 0 | mot_auto > 0")

fig = plt.figure(figsize=(9, 3))
outer = gridspec.GridSpec(nrows=1, ncols=3)
for n, t in enumerate(df_mot.groupby("country")):
    country, tmp = t
    inner = gridspec.GridSpecFromSubplotSpec(
        nrows=6, ncols=6, subplot_spec=outer[n], wspace=-0.2, hspace=-0.3
    )
    for i in range(6):
        ax_main = plt.Subplot(fig, inner[0:4, 2:6])
        ax_main.plot(
            tmp["Component A"], tmp["Component B"], ".", color=COLORS["yellow"]
        )
        ax_main.set_xlim(-0.6, 0.6)
        ax_main.set_ylim(-0.6, 0.6)
        for spine in ax_main.spines:
            if spine not in ["left", "bottom"]:
                ax_main.spines[spine].set_visible(False)
        fig.add_subplot(ax_main)
        ax_main.title.set_text(country.upper())

        ax = plt.Subplot(fig, inner[5, 2:6], sharex=ax_main)
        ax.set_xlim(-0.6, 0.6)
        ax.boxplot(
            tmp["Component A"],
            patch_artist=True,
            medianprops=dict(linestyle="-", linewidth=1, color="black"),
            widths=0.5,
            boxprops={"facecolor": COLORS["yellow"]},
            orientation="horizontal",
        )
        for spine in ax.spines:
            ax.spines[spine].set_visible(False)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        fig.add_subplot(ax)

        ax = plt.Subplot(fig, inner[0:4, 0], sharey=ax_main)
        ax.boxplot(
            tmp["Component B"],
            patch_artist=True,
            medianprops=dict(linestyle="-", linewidth=1, color="black"),
            widths=0.5,
            boxprops={"facecolor": COLORS["yellow"]},
        )
        for spine in ax.spines:
            ax.spines[spine].set_visible(False)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        fig.add_subplot(ax)

plt.tight_layout()
plt.show()
fig.savefig(PNG / "motivation_semantic_spread.png", dpi=200)
# %%
## CAPABILITIES
df_cap = df.query("cap_psychological > 0 | cap_physical > 0")

fig = plt.figure(figsize=(9, 3))
outer = gridspec.GridSpec(nrows=1, ncols=3)
for n, t in enumerate(df_cap.groupby("country")):
    country, tmp = t
    inner = gridspec.GridSpecFromSubplotSpec(
        nrows=6, ncols=6, subplot_spec=outer[n], wspace=-0.2, hspace=-0.3
    )
    for i in range(6):
        ax_main = plt.Subplot(fig, inner[0:4, 2:6])
        ax_main.plot(tmp["Component A"], tmp["Component B"], ".", color=COLORS["blue"])
        ax_main.set_xlim(-0.6, 0.6)
        ax_main.set_ylim(-0.6, 0.6)
        for spine in ax_main.spines:
            if spine not in ["left", "bottom"]:
                ax_main.spines[spine].set_visible(False)
        fig.add_subplot(ax_main)
        ax_main.title.set_text(country.upper())

        ax = plt.Subplot(fig, inner[5, 2:6], sharex=ax_main)
        ax.set_xlim(-0.6, 0.6)
        ax.boxplot(
            tmp["Component A"],
            patch_artist=True,
            medianprops=dict(linestyle="-", linewidth=1, color="black"),
            widths=0.5,
            boxprops={"facecolor": COLORS["blue"]},
            orientation="horizontal",
        )
        for spine in ax.spines:
            ax.spines[spine].set_visible(False)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        fig.add_subplot(ax)

        ax = plt.Subplot(fig, inner[0:4, 0], sharey=ax_main)
        ax.boxplot(
            tmp["Component B"],
            patch_artist=True,
            medianprops=dict(linestyle="-", linewidth=1, color="black"),
            widths=0.5,
            boxprops={"facecolor": COLORS["blue"]},
        )
        for spine in ax.spines:
            ax.spines[spine].set_visible(False)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        fig.add_subplot(ax)

plt.tight_layout()
plt.show()
fig.savefig(PNG / "capabilities_sematic_spread.png", dpi=200)
# %%
## OPPORTUNITIES
df_opp = df.query("opp_physical > 0 | opp_social > 0")

fig = plt.figure(figsize=(9, 3))
outer = gridspec.GridSpec(nrows=1, ncols=3)
for n, t in enumerate(df_opp.groupby("country")):
    country, tmp = t
    inner = gridspec.GridSpecFromSubplotSpec(
        nrows=6, ncols=6, subplot_spec=outer[n], wspace=-0.2, hspace=-0.3
    )
    for i in range(6):
        ax_main = plt.Subplot(fig, inner[0:4, 2:6])
        ax_main.plot(tmp["Component A"], tmp["Component B"], ".", color=COLORS["green"])
        ax_main.set_xlim(-0.6, 0.6)
        ax_main.set_ylim(-0.6, 0.6)
        for spine in ax_main.spines:
            if spine not in ["left", "bottom"]:
                ax_main.spines[spine].set_visible(False)
        fig.add_subplot(ax_main)
        ax_main.title.set_text(country.upper())

        ax = plt.Subplot(fig, inner[5, 2:6], sharex=ax_main)
        ax.set_xlim(-0.6, 0.6)
        ax.boxplot(
            tmp["Component A"],
            patch_artist=True,
            medianprops=dict(linestyle="-", linewidth=1, color="black"),
            widths=0.5,
            boxprops={"facecolor": COLORS["green"]},
            orientation="horizontal",
        )
        for spine in ax.spines:
            ax.spines[spine].set_visible(False)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        fig.add_subplot(ax)

        ax = plt.Subplot(fig, inner[0:4, 0], sharey=ax_main)
        ax.boxplot(
            tmp["Component B"],
            patch_artist=True,
            medianprops=dict(linestyle="-", linewidth=1, color="black"),
            widths=0.5,
            boxprops={"facecolor": COLORS["green"]},
        )
        for spine in ax.spines:
            ax.spines[spine].set_visible(False)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        fig.add_subplot(ax)

plt.tight_layout()
plt.show()
fig.savefig(PNG / "opportunities_semantic_spread.png", dpi=200)
# %%
