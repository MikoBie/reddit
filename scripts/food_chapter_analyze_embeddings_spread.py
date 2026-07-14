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
## Define functions
def plot_scatter_with_boxplots(
    df: pd.DataFrame, color: str = COLORS["yellow"]
) -> plt.Figure:
    """Plots a scatter plot with boxplots that show distribution of the variables.

    Parameters
    ----------
    df
        a data frame with Component A and Component B columns.
    color, optional
        color of the plots, by default COLORS["yellow"]

    Returns
    -------
        a series of three scatter plots with boxplots.
    """
    fig = plt.Figure(figsize=(9, 3))
    outer = gridspec.GridSpec(nrows=1, ncols=3)
    for n, t in enumerate(df.groupby("country")):
        country, tmp = t
        nrows = len(tmp)
        inner = gridspec.GridSpecFromSubplotSpec(
            nrows=6, ncols=6, subplot_spec=outer[n], wspace=0.1, hspace=0.2
        )
        ax_main = plt.Subplot(fig, inner[1:6, 0:5])
        ax_main.plot(tmp["Component A"], tmp["Component B"], ".", color=color)
        ax_main.set_xlim(-0.6, 0.6)
        ax_main.set_xlabel("Component A")
        ax_main.set_ylim(-0.6, 0.6)
        ax_main.set_ylabel("Component B")
        ax_main.tick_params("x", top=True, labeltop=False)
        ax_main.tick_params("y", right=True, labelright=False)
        fig.add_subplot(ax_main)

        ax = plt.Subplot(fig, inner[0, 0:5], sharex=ax_main)
        ax.set_xlim(-0.6, 0.6)
        ax.boxplot(
            tmp["Component A"],
            patch_artist=True,
            medianprops=dict(linestyle="-", linewidth=1, color="black"),
            widths=0.5,
            boxprops={"facecolor": color},
            orientation="horizontal",
        )
        for spine in ax.spines:
            ax.spines[spine].set_visible(False)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.title.set_text(country.upper() + "\n" + f"(n = {nrows})")
        fig.add_subplot(ax)

        ax = plt.Subplot(fig, inner[1:6, 5], sharey=ax_main)
        ax.boxplot(
            tmp["Component B"],
            patch_artist=True,
            medianprops=dict(linestyle="-", linewidth=1, color="black"),
            widths=0.5,
            boxprops={"facecolor": color},
        )
        for spine in ax.spines:
            ax.spines[spine].set_visible(False)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        fig.add_subplot(ax)
    return fig


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

fig = plot_scatter_with_boxplots(df=df_mot, color=COLORS["yellow"])
fig.tight_layout()
fig.savefig(PNG / "motivation_semantic_spread.png", dpi=200)
fig
# %%
## CAPABILITIES
df_cap = df.query("cap_psychological > 0 | cap_physical > 0")

fig = plot_scatter_with_boxplots(df=df_cap, color=COLORS["blue"])
fig.tight_layout()
fig.savefig(PNG / "capabilities_sematic_spread.png", dpi=200)
fig
# %%
## OPPORTUNITIES
df_opp = df.query("opp_physical > 0 | opp_social > 0")

fig = plot_scatter_with_boxplots(df=df_opp, color=COLORS["green"])
fig.tight_layout()
fig.savefig(PNG / "opportunities_semantic_spread.png", dpi=200)
fig

# %%
